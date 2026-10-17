"""Power of the refinement rounds, and what correcting it changes.

Block power control scales shell k of a message by a gain g_k computed over all
T tokens of the message, so that the shell has unit average power per symbol.
The stream simulation (dnsc/online.py, dnsc/refine.py) transmits every message
once with these gains and reads the refinement rounds off that transmission. A
refinement round, however, sends shell k only for the requested tokens; with the
gain of the whole message its average power is

    P_k = g_k^2 * sum_{t in R_k} |c_t,k|^2 / (|R_k| d_k),

which is one only on average. A transmitter that meets the power constraint in
every round must use the gain of the requested subset, g'_k, instead; the
received symbol is then c + (y - c) g_k / g'_k with precision g'_k^2 SNR (the same
noise realisation, rescaled).

This script measures P_k and compares the request policy with the two gains on
held-out sentences (AG News and TeleQnA test splits), teacher-forced as in the
stream runner, with the published codebook and the stream's first round of 6
symbols, for the frozen GPT-2 prior without context and with the true preceding
sentences of the document as context. Fixed prefixes send every shell to every
token and are unaffected.

    python scripts/check_refine_power.py --messages 600 --out results/refine_power
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dnsc.channel import Channel, capacity_real, shell_slices
from dnsc.codec import load_codec
from dnsc.config import load_config
from dnsc.data import get_corpus
from dnsc.linkbudget import symbols_at_accuracy
from dnsc.utils import environment, set_seed

ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


def h2(p: float) -> float:
    return 0.0 if p <= 0.0 or p >= 1.0 else -p * math.log2(p) - (1 - p) * math.log2(1 - p)


class Tally:
    def __init__(self, k: int):
        self.tokens = 0
        self.correct = 0
        self.symbols = 0.0
        self.open = np.zeros(k)
        self.cont = np.zeros(k)
        self.blocks: list[tuple[int, float]] = []  # (symbols in the block, power of the block with the message gain)

    def point(self, capacity: float) -> dict:
        ent = sum(n * h2(c / n) for n, c in zip(self.open, self.cont) if n > 0) / self.tokens
        bm = float(self.open.sum()) / self.tokens
        fwd = self.symbols / self.tokens
        return {"tok_acc": self.correct / self.tokens, "forward": fwd, "total_entropy": fwd + ent / capacity, "total_bitmap": fwd + bm / capacity}


def policy(codec, ids, y, prec, gains_tok, owner, log_prior, tau, first, corrected, tally):
    """Selective refinement, teacher-forced, round by round; returns nothing, accumulates in tally."""
    code = codec.codebook().to(y.dtype)
    slices = shell_slices(codec.grans)
    scale = float(codec.lik_weight)
    score = log_prior.clone()
    for sl in slices[: first + 1]:  # round 1: these shells reach every token (unit power exactly)
        w = prec[:, sl]
        score += scale * ((w * y[:, sl]) @ code[:, sl].t() - 0.5 * w @ (code[:, sl] ** 2).t())
    post = F.log_softmax(score, dim=1)
    conf, decided = post.max(dim=1)
    conf = conf.exp()
    level = torch.full_like(ids, first)
    n_levels = len(codec.grans)
    is_open = torch.ones_like(ids, dtype=torch.bool) if first < n_levels - 1 else torch.zeros_like(ids, dtype=torch.bool)
    for k in range(first + 1, n_levels):
        request = is_open & (conf < tau)
        tally.open[k - 1] += float(is_open.sum())
        tally.cont[k - 1] += float(request.sum())
        if not bool(request.any()):
            break
        sl = slices[k]
        width = sl.stop - sl.start
        true_c = code[ids, sl]  # (N, d_k), known to the transmitter
        g_msg = gains_tok[:, sl.start]
        g_use = g_msg.clone()
        for b in torch.unique(owner[request]).tolist():
            mine = request & (owner == b)
            energy = float(true_c[mine].pow(2).sum())
            n = int(mine.sum())
            power = float(g_msg[mine][0]) ** 2 * energy / (n * width)
            tally.blocks.append((n * width, power))
            if corrected:
                g_use[mine] = math.sqrt(n * width / max(energy, 1e-12))
        rows = request.nonzero().squeeze(1)
        ratio = (g_msg[rows] / g_use[rows]).unsqueeze(1)
        y_k = true_c[rows] + (y[rows][:, sl] - true_c[rows]) * ratio
        w_k = prec[rows][:, sl] * (g_use[rows] / g_msg[rows]).pow(2).unsqueeze(1)
        score[rows] += scale * ((w_k * y_k) @ code[:, sl].t() - 0.5 * w_k @ (code[:, sl] ** 2).t())
        post_r = F.log_softmax(score[rows], dim=1)
        c_r, d_r = post_r.max(dim=1)
        conf[rows], decided[rows] = c_r.exp(), d_r
        level[rows] = k
        is_open = request & (level < n_levels - 1)
    sizes = torch.tensor(codec.grans, dtype=torch.float32)
    tally.tokens += int(ids.numel())
    tally.correct += int((decided == ids).sum())
    tally.symbols += float(sizes[level].sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/gpt2.yaml")
    parser.add_argument("--checkpoint", default="results/gpt2/base")
    parser.add_argument("--messages", type=int, default=600)
    parser.add_argument("--out", default="results/refine_power")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    set_seed(args.seed)
    cfg = load_config(ROOT / args.config)
    codec = load_codec(cfg, ROOT / args.checkpoint, torch.device("cpu"))
    codec.eval()
    codec.set_slot(None)
    rng = np.random.default_rng(args.seed)
    # one sentence per document, at a random position, with the sentences before it
    items = []
    for name in cfg["stream"]["domains"]:
        for doc in get_corpus(name, cfg["data"]).documents("test"):
            texts = [r["text"] for r in doc]
            i = int(rng.integers(len(texts)))
            items.append((texts[i], texts[:i]))
    items = [items[i] for i in rng.permutation(len(items))[: args.messages]]
    first = codec.grans.index(int(cfg["decoder"]["start_m"]))
    capacity = capacity_real(float(cfg["stream"]["snr"]["mean"]))
    snr_cfg = cfg["stream"]["snr"]
    snrs = np.clip(rng.normal(snr_cfg["mean"], snr_cfg["sigma"], size=len(items)), snr_cfg["min"], snr_cfg["max"])  # one per message, as in the decoder selection
    results = {}
    for variant in ("no_context", "true_context"):
        channel = Channel("awgn", seed=args.seed + 7)  # the same noise draws for both variants
        tallies = {(tau, corr): Tally(len(codec.grans)) for tau in THRESHOLDS for corr in (False, True)}
        fixed_hits = np.zeros(len(codec.grans))
        n_tok = 0
        with torch.no_grad():
            for start in range(0, len(items), 6):
                chunk = items[start : start + 6]
                texts = [t for t, _ in chunk]
                if variant == "true_context":
                    continued = [len(h) > 0 for _, h in chunk]
                    contexts = [codec.history_ids(h, int(cfg["context"]["max_tokens"])) for _, h in chunk]
                else:
                    continued, contexts = None, None
                batch = codec.tokenize(texts, continued)
                x = codec.encode(batch)
                gains = codec.block_gains(x, batch.lens)
                y, prec = codec.transmit(x, batch.lens, torch.tensor(snrs[start : start + len(chunk)], dtype=torch.float32), channel)
                ids, y_flat, w_flat = codec.flat_channel(batch, y, prec)
                lens = batch.lens.tolist()
                owner = torch.repeat_interleave(torch.arange(len(lens)), torch.tensor(lens))
                gains_tok = gains[owner]
                log_prior = codec._adjust_prior(codec.lm_logits_tf(batch, contexts, flat=True), None)
                # fixed prefixes (every token gets every shell up to m: exact power)
                code = codec.codebook()
                score = log_prior.clone()
                for k, sl in enumerate(shell_slices(codec.grans)):
                    w = w_flat[:, sl]
                    score += float(codec.lik_weight) * ((w * y_flat[:, sl]) @ code[:, sl].t() - 0.5 * w @ (code[:, sl] ** 2).t())
                    fixed_hits[k] += float((score.argmax(1) == ids).sum())
                n_tok += int(ids.numel())
                for tau in THRESHOLDS:
                    for corr in (False, True):
                        policy(codec, ids, y_flat, w_flat, gains_tok, owner, log_prior, tau, first, corr, tallies[(tau, corr)])
        fixed_acc = (fixed_hits / n_tok).tolist()
        fixed_at = symbols_at_accuracy([float(m) for m in codec.grans], fixed_acc, 0.9)
        out = {"tokens": n_tok, "fixed_curve": [{"m": m, "tok_acc": a} for m, a in zip(codec.grans, fixed_acc)], "fixed_at_target": fixed_at}
        for corr, name in ((False, "message_gain"), (True, "subset_gain")):
            curve = [{"threshold": tau, **tallies[(tau, corr)].point(capacity)} for tau in THRESHOLDS]
            reading = {key: symbols_at_accuracy([p[key] for p in curve], [p["tok_acc"] for p in curve], 0.9) for key in ("forward", "total_entropy", "total_bitmap")}
            blocks = [b for tau in THRESHOLDS for b in ([] if corr else tallies[(tau, False)].blocks)]
            out[name] = {"curve": curve, "at_target": reading, "vs_fixed_entropy_pct": 100 * (reading["total_entropy"] / fixed_at - 1)}
        # power of the refinement blocks with the message gain, at the stream's operating threshold
        blocks = np.array(tallies[(0.9, False)].blocks)
        out["power_at_tau_0.9"] = {"blocks": int(len(blocks)), "symbol_weighted_mean": float((blocks[:, 0] * blocks[:, 1]).sum() / blocks[:, 0].sum()), "median": float(np.median(blocks[:, 1])), "p5": float(np.percentile(blocks[:, 1], 5)), "p95": float(np.percentile(blocks[:, 1], 95))}
        results[variant] = out
        print(variant, json.dumps({k: out[k] for k in ("fixed_at_target", "power_at_tau_0.9")}), {n: out[n]["at_target"] for n in ("message_gain", "subset_gain")})
    target = ROOT / args.out
    target.mkdir(parents=True, exist_ok=True)
    summary = {"messages": len(items), "first_round_symbols": codec.grans[first], "thresholds": THRESHOLDS, "capacity": capacity, "results": results, "env": environment()}
    (target / "refine_power.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    lines = [
        "**Refinement rounds with the gain of the whole message (as simulated on the streams) and with the gain of the requested tokens (power exactly one in every round): held-out AG News and TeleQnA sentences, frozen GPT-2 prior, first round of 6 symbols, teacher-forced request policy, AWGN with the SNR law of the stream; real symbols per token for 90% token accuracy, signaling at its entropy at the capacity of 10 dB**",
        "",
        "| prior | messages | tokens | refinement-block power with the message gain at threshold 0.9 (symbol-weighted mean; 5th-95th percentile of blocks) | fixed prefix | refinement, message gain | refinement, subset gain | change from correcting the power |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for variant, out in results.items():
        p = out["power_at_tau_0.9"]
        a, b = out["message_gain"]["at_target"]["total_entropy"], out["subset_gain"]["at_target"]["total_entropy"]
        lines.append(f"| {variant} | {len(items)} | {out['tokens']} | {p['symbol_weighted_mean']:.3f} ({p['p5']:.2f}-{p['p95']:.2f}) | {out['fixed_at_target']:.3f} | {a:.3f} ({out['message_gain']['vs_fixed_entropy_pct']:+.1f}%) | {b:.3f} ({out['subset_gain']['vs_fixed_entropy_pct']:+.1f}%) | {100 * (b / a - 1):+.1f}% |")
    (target / "refine_power.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
