"""How much is there to gain from each memory of the receiver? (diagnostic)

Before any long stream run, this measures on held-out text of every domain
what each memory could buy at best, with oracle inputs:

frozen            every sentence decoded on its own
context           the true earlier sentences of the same document in front
context+bias      context, after a short adaptation of the medium memory
context+bias+lora context, after the same adaptation of both gradient memories

For the adapted variants it also reports what the adaptation did to the
*other* domains (negative transfer): this is what a single adapter suffers
when domains alternate, and what separate slots avoid.

Reported per variant: the cross-entropy of the prior alone (bits/token) and
the posterior cross-entropy and token accuracy at the reference prefix length
and the mean stream SNR. These are upper references (true text as context, no
staleness, no synchronisation cost), not results of the method.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .channel import Channel
from .codec import SemanticCodec, load_codec
from .data import get_corpus
from .data.stream import _probes_from
from .train import evaluate_tf, sample_grans
from .utils import Timer, environment, get_device, save_json, set_seed


def _measure(codec: SemanticCodec, probes, with_context: bool, m: int, snr_db: float, batch_size: int, limit: int) -> dict:
    texts = [p.text for p in probes]
    flags = [bool(p.history) for p in probes] if with_context else None
    contexts = [codec.history_ids(p.history, limit) for p in probes] if with_context else None
    out = evaluate_tf(codec, texts, m, snr_db, Channel("awgn", 9001), batch_size=batch_size, contexts=contexts, continued=flags)
    out["with_history"] = int(sum(bool(p.history) for p in probes))
    return out


def _adapt(codec: SemanticCodec, docs: list[list[str]], groups: tuple[str, ...], cfg, steps: int, seed: int) -> None:
    """A short burst of the stream's own adaptation (same loss, learning rates
    and context), without synchronisation delay."""
    named = codec.groups()
    lrs = {"rx_bias": float(cfg["nested"]["lrs"]["medium"]), "rx_lora": float(cfg["nested"]["lrs"]["slow"])}
    param_groups = [{"params": [p for _, p in named[g]], "lr": lrs[g]} for g in groups]
    params = [p for group in param_groups for p in group["params"]]
    for p in params:
        p.requires_grad_(True)
    optimizer = torch.optim.Adam(param_groups)
    rng = np.random.default_rng(seed)
    channel = Channel(cfg["channel"]["type"], seed=seed + 1)
    scfg = cfg["stream"]
    m_op, limit = int(scfg["m_op"]), int(cfg["context"]["max_tokens"])
    positions = [(d, i) for d, doc in enumerate(docs) for i in range(len(doc))]
    for _ in range(steps):
        picks = [positions[int(j)] for j in rng.integers(0, len(positions), size=int(scfg["batch_size"]))]
        batch = codec.tokenize([docs[d][i] for d, i in picks], [i > 0 for _, i in picks])
        contexts = [codec.history_ids(docs[d][:i], limit) for d, i in picks]
        snr = float(rng.uniform(float(scfg["snr"]["min"]), float(scfg["snr"]["max"])))
        with torch.no_grad():
            y, prec = codec.transmit(codec.encode(batch), batch.lens, snr, channel)
            ids, y_flat, w_flat = codec.flat_channel(batch, y, prec)
        with torch.set_grad_enabled("rx_lora" in groups):
            logits = codec.lm_logits_tf(batch, contexts, flat=True)
        log_prior = codec._adjust_prior(logits, None)
        grans = [m_op] + sample_grans([m for m in codec.grans if m != m_op], int(scfg["extra_grans"]), rng)
        loss = sum(codec.flat_loss(ids, y_flat, w_flat, m, log_prior) for m in grans) / len(grans)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, float(cfg["nested"]["grad_clip"]))
        optimizer.step()
    codec.freeze_all()


def run_headroom(cfg, ckpt_dir: str | Path, out_path: str | Path, n: int = 64, steps: int = 150) -> dict:
    set_seed(int(cfg["seed"]))
    device = get_device(cfg["device"])
    codec = load_codec(cfg, ckpt_dir, device)
    codec.freeze_all()
    codec.eval()
    scfg = cfg["stream"]
    m_op, snr = int(scfg["m_op"]), float(scfg["snr"]["mean"])
    batch_size, limit = int(cfg["eval"]["batch_size"]), int(cfg["context"]["max_tokens"])
    adapt_domains = list(dict.fromkeys(list(scfg["domains"]) + [scfg["temporal_corpus"]]))
    names = list(dict.fromkeys(list(cfg["data"]["base"]) + adapt_domains))
    rng = np.random.default_rng(int(cfg["seed"]) + 5)
    probes, train_docs = {}, {}
    for name in names:
        corpus = get_corpus(name, cfg["data"])
        probes[name] = _probes_from(corpus.documents("test"), n, rng)
        train_docs[name] = [[r["text"] for r in doc] for doc in corpus.documents("train")]
    pristine = {key: value.clone() for key, value in codec.codec_state().items() if not key.startswith("lm.") or "lora_" in key}
    timer = Timer()

    rows = []
    for name in names:
        for variant, with_context in (("frozen", False), ("context", True)):
            rows.append({"domain": name, "variant": variant, "measured_on": name, **_measure(codec, probes[name], with_context, m_op, snr, batch_size, limit)})
        print(f"[headroom] {name}: frozen prior and context measured ({timer.elapsed() / 60:.1f} min)", flush=True)
    for name in adapt_domains:
        for label, groups in (("context+bias", ("rx_bias",)), ("context+bias+lora", ("rx_bias", "rx_lora"))):
            _adapt(codec, train_docs[name], groups, cfg, steps, seed=int(cfg["seed"]) + 17)
            for other in names:  # the adapted domain itself, and negative transfer to the others
                rows.append({"domain": name, "variant": label, "measured_on": other, "steps": steps, **_measure(codec, probes[other], True, m_op, snr, batch_size, limit)})
            codec.load_state_dict(pristine, strict=False)
            codec.invalidate_codebook()
            print(f"[headroom] {name}: '{label}' adapted for {steps} steps and measured ({timer.elapsed() / 60:.1f} min)", flush=True)

    result = {"m_op": m_op, "snr_db": snr, "n": n, "steps": steps, "rows": rows, "config": cfg.to_dict(), "env": environment()}
    save_json(out_path, result)
    return result


def format_rows(result: dict) -> list[dict]:
    """One line per (domain, memory) with the change against decoding every
    sentence on its own."""
    frozen = {row["measured_on"]: row for row in result["rows"] if row["variant"] == "frozen"}
    out = []
    for row in result["rows"]:
        reference = frozen[row["measured_on"]]
        adapted = row["variant"].startswith("context+")
        first = row["variant"] == "frozen"
        out.append(
            {
                "measured on": row["measured_on"],
                "memory": row["variant"] + (f" (adapted on {row['domain']})" if adapted else ""),
                "n": row["n"],
                "prior bits/token": f"{row['prior_bits']:.3f}",
                "Δ prior": "-" if first else f"{row['prior_bits'] - reference['prior_bits']:+.3f}",
                f"posterior bits @m={result['m_op']}": f"{row['ce_bits']:.3f}",
                "Δ posterior": "-" if first else f"{row['ce_bits'] - reference['ce_bits']:+.3f}",
                "token acc": f"{row['tok_acc']:.3f}",
                "Δ acc": "-" if first else f"{row['tok_acc'] - reference['tok_acc']:+.3f}",
            }
        )
    return sorted(out, key=lambda r: r["measured on"])
