"""Choose the decoder of selective refinement on validation data only.

    python scripts/tune_decoder.py --config configs/gpt2.yaml

The candidates, the validation set and the selection rule are fixed in
``docs/METHOD.md`` (section 2.5, "Decoder of the refinement"): first shell of
2, 4 or 6 symbols x reopening (stage 1, greedy), then beam widths 2 and 4 for
the winner (stage 2). Every candidate decodes exactly the same received
symbols, free-running, and is scored by the real symbols per token it needs
for the target token accuracy with the request signalling charged at its
entropy. The fixed-prefix reference is decoded with the same beam as the
candidate it is compared with.

Writes ``<out_dir>/eval/decoder_selection.json`` (every operating point of
every candidate, and the decision) and prints the tables. Finished stages are
kept in the file, so an interrupted run continues where it stopped.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch

from dnsc.channel import Channel, capacity_real
from dnsc.cli import make_parser, parse
from dnsc.codec import load_codec
from dnsc.data import get_corpus
from dnsc.linkbudget import symbols_at_accuracy
from dnsc.refine import FeedbackMeter, progressive_decode
from dnsc.utils import environment, get_device, load_json, save_json, set_seed

THRESHOLDS = [0.7, 0.85, 0.95, 0.99]
STAGE1 = [(1, start, reopen) for start in (0, 1, 2) for reopen in (False, True)]  # (beam, first shell index, reopen)
BEAMS = (2, 4)
N_DOCS = 32  # per corpus
BATCH = 6


def finite(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def validation_messages(cfg, codec, rng: np.random.Generator) -> list[dict]:
    """One sentence at a random position of N_DOCS validation documents of
    each stream corpus, with the true preceding sentences as context."""
    names = list(dict.fromkeys(list(cfg["stream"]["domains"]) + [cfg["stream"]["temporal_corpus"]]))
    limit = int(cfg["context"]["max_tokens"])
    out = []
    for name in names:
        docs = get_corpus(name, cfg["data"]).documents("val")
        for d in rng.permutation(len(docs))[:N_DOCS]:
            doc = docs[int(d)]
            i = int(rng.integers(len(doc)))
            history = [r["text"] for r in doc[:i]]
            out.append({"corpus": name, "text": doc[i]["text"], "continued": i > 0, "context": codec.history_ids(history, limit)})
    return out


def key(beam: int, start: int, reopen: bool) -> str:
    return f"beam{beam}_start{start}_{'reopen' if reopen else 'closed'}"


def main() -> None:
    parser = make_parser(__doc__)
    parser.add_argument("--ckpt", default="", help="checkpoint directory (default: <out_dir>/base)")
    cfg, args = parse(parser)
    set_seed(int(cfg["seed"]))
    device = get_device(cfg["device"])
    ckpt = Path(args.ckpt) if args.ckpt else Path(cfg["out_dir"]) / "base"
    codec = load_codec(cfg, ckpt, device)
    codec.freeze_all()
    codec.eval()
    grans = codec.grans
    target = float(cfg["stream"]["target_acc"])
    rng = np.random.default_rng(20261015)
    messages = validation_messages(cfg, codec, rng)
    # one SNR per message, drawn independently from the stationary law of the stream's SNR process
    # (an AR(1) process with this mean and standard deviation, clipped to its range)
    scfg = cfg["stream"]["snr"]
    snrs = np.clip(np.random.default_rng(20261016).normal(float(scfg["mean"]), float(scfg["sigma"]), size=len(messages)), float(scfg["min"]), float(scfg["max"]))
    capacity = capacity_real(float(cfg["stream"]["snr"]["mean"]))
    out_path = Path(cfg["out_dir"]) / "eval" / "decoder_selection.json"
    state = load_json(out_path) if out_path.exists() else {"candidates": {}, "fixed": {}}
    print(f"[tune] {len(messages)} validation messages, SNR mean {snrs.mean():.2f} dB (sd {snrs.std():.2f}), thresholds {THRESHOLDS}, target {target:.0%}")

    # transmit every message once; every candidate decodes these same symbols
    channel = Channel("awgn", seed=20261017)
    batches = []
    for i in range(0, len(messages), BATCH):
        chunk = messages[i : i + BATCH]
        batch = codec.tokenize([m["text"] for m in chunk], [m["continued"] for m in chunk])
        with torch.no_grad():
            y, prec = codec.transmit(codec.encode(batch), batch.lens, torch.tensor(snrs[i : i + BATCH], dtype=torch.float32), channel)
        batches.append((batch, y, prec, [m["context"] for m in chunk]))
    n_tokens = sum(int(b.lens.sum()) for b, _, _, _ in batches)

    def hits(batch, rows) -> int:
        return sum(int(a == b) for row, length, got in zip(batch.ids.tolist(), batch.lens.tolist(), rows) for a, b in zip(row[:length], got))

    def fixed_curve(beam: int) -> list[dict]:
        name = f"beam{beam}"
        if name not in state["fixed"]:
            t0 = time.time()
            correct = np.zeros(len(grans))
            for batch, y, prec, ctx in batches:
                for k, m in enumerate(grans):
                    correct[k] += hits(batch, codec.decode(y, m, prec, batch.lens, contexts=ctx, beam=beam).token_ids)
            state["fixed"][name] = [{"m": m, "tok_acc": float(c / n_tokens)} for m, c in zip(grans, correct)]
            save_json(out_path, state)
            print(f"[tune] fixed prefix, beam {beam}: " + "  ".join(f"m={p['m']}:{p['tok_acc']:.3f}" for p in state["fixed"][name]) + f"  ({time.time() - t0:.0f}s)", flush=True)
        return state["fixed"][name]

    def evaluate(beam: int, start: int, reopen: bool) -> dict:
        name = key(beam, start, reopen)
        if name in state["candidates"]:
            return state["candidates"][name]
        t0 = time.time()
        sums = np.zeros((len(THRESHOLDS), 3))  # symbols x tokens, correct, rounds x messages
        meters = [FeedbackMeter(len(grans)) for _ in THRESHOLDS]
        n_messages = 0
        for batch, y, prec, ctx in batches:
            first = None
            for j, tau in enumerate(THRESHOLDS):
                out = progressive_decode(codec, y, prec, batch.lens, tau, ctx, start=start, first_pass=first, beam=beam, reopen=reopen)
                first = out["first_pass"]
                sums[j] += (out["symbols"] * out["n_tokens"], hits(batch, out["token_ids"]), out["rounds"] * len(batch.lens))
                meters[j].merge(out["meter"])
            n_messages += len(batch.lens)
        points = []
        for j, tau in enumerate(THRESHOLDS):
            symbols, accuracy, rounds = sums[j][0] / n_tokens, sums[j][1] / n_tokens, sums[j][2] / n_messages
            bitmap, entropy = meters[j].bitmap_bits(), meters[j].entropy_bits()
            points.append({"threshold": tau, "symbols": symbols, "tok_acc": accuracy, "rounds": rounds, "feedback_bits": bitmap, "feedback_entropy_bits": entropy, "symbols_total": symbols + bitmap / capacity, "symbols_total_entropy": symbols + entropy / capacity})
        acc = [p["tok_acc"] for p in points]
        fixed = fixed_curve(beam)
        result = {
            "beam": beam, "start_m": grans[start], "reopen": reopen, "points": points,
            "forward": finite(symbols_at_accuracy([p["symbols"] for p in points], acc, target)),
            "total_entropy": finite(symbols_at_accuracy([p["symbols_total_entropy"] for p in points], acc, target)),
            "total_bitmap": finite(symbols_at_accuracy([p["symbols_total"] for p in points], acc, target)),
            "fixed": finite(symbols_at_accuracy([float(p["m"]) for p in fixed], [p["tok_acc"] for p in fixed], target)),
            "seconds": time.time() - t0,
        }
        state["candidates"][name] = result
        save_json(out_path, state)
        show = lambda v: "not reached" if v is None else f"{v:.3f}"  # noqa: E731
        print(f"[tune] {name:<26} forward {show(result['forward'])}  +entropy {show(result['total_entropy'])}  +bitmap {show(result['total_bitmap'])}  fixed (same beam) {show(result['fixed'])}  ({result['seconds']:.0f}s)", flush=True)
        return result

    def criterion(result: dict) -> float:
        return math.inf if result["total_entropy"] is None else result["total_entropy"]

    def best(results: list[dict]) -> dict:
        """Lowest criterion; within 1 % of it, the fewest rounds per message at the operating threshold."""
        top = min(criterion(r) for r in results)
        tied = [r for r in results if criterion(r) <= top * 1.01]
        rounds = lambda r: next(p["rounds"] for p in r["points"] if p["threshold"] == 0.95)  # noqa: E731
        return min(tied, key=lambda r: (rounds(r), criterion(r)))

    fixed_curve(1)
    stage1 = [evaluate(*c) for c in STAGE1]
    winner = best(stage1)
    start = grans.index(winner["start_m"])
    stage2 = [evaluate(beam, start, winner["reopen"]) for beam in BEAMS]
    chosen = winner
    for result in sorted(stage2, key=criterion):
        if criterion(result) < criterion(winner) * 0.98:
            chosen = result
            break
    state["decision"] = {
        "stage1_winner": key(1, start, winner["reopen"]),
        "chosen": key(chosen["beam"], start, chosen["reopen"]),
        "beam": chosen["beam"], "start_m": chosen["start_m"], "reopen": chosen["reopen"],
        "criterion": criterion(chosen), "fixed_same_beam": chosen["fixed"], "fixed_greedy": finite(symbols_at_accuracy([float(p["m"]) for p in state["fixed"]["beam1"]], [p["tok_acc"] for p in state["fixed"]["beam1"]], target)),
        "baseline_v03": criterion(state["candidates"][key(1, 0, False)]),
    }
    state.update({"n_messages": len(messages), "n_tokens": n_tokens, "snr_mean": float(snrs.mean()), "snr_sd": float(snrs.std()), "thresholds": THRESHOLDS, "target_acc": target, "capacity": capacity, "config": cfg.to_dict(), "env": environment()})
    save_json(out_path, state)
    d = state["decision"]
    print(f"\n[tune] decision: {d['chosen']}  ->  {d['criterion']:.3f} symbols/token at {target:.0%} with signalling (version 0.3 decoder: {d['baseline_v03']:.3f}); fixed prefix with the same beam {d['fixed_same_beam']}, greedy {d['fixed_greedy']}")
    print(f"[tune] wrote {out_path}")


if __name__ == "__main__":
    main()
