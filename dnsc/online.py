"""Online adaptation on a multi-timescale drift stream (prequential protocol).

Every incoming batch is first *transmitted and scored* with the codec as it
stands, and only then used for adaptation ("test-then-train"), so the reported
stream metrics are never contaminated by training on the scored data.

Methods
-------
frozen            no adaptation; zero synchronisation cost
tx_only           transmitter-side levels only; zero synchronisation cost
single_rate       every level at one period and one learning rate, with the
                  period chosen so that its synchronisation budget equals the
                  nested method's (the matched ablation)
ewc               single_rate plus an Elastic-Weight-Consolidation penalty
full_online       every level at every step (adaptivity upper reference;
                  synchronisation cost is prohibitive by construction)
nested            the proposed three-level schedule with an EMA anchor
nested_no_ema     nested without the continuum-memory anchor
nested_two_level  receiver levels collapsed to one period (matched budget)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from .channel import Channel
from .codec import SemanticCodec, load_codec
from .data import build_stream, get_corpus
from .data.stream import build_probes
from .linkbudget import net_semantic_rate
from .metrics import load_similarity
from .nested import Level, NestedOptimizer
from .train import LN2, evaluate_gen, evaluate_tf, sample_grans
from .utils import Timer, environment, get_device, save_json, set_seed

METHODS = ("frozen", "tx_only", "single_rate", "ewc", "full_online", "nested", "nested_no_ema", "nested_two_level")


def _dense_bits(params, bits: int, topk: float) -> int:
    total = 0
    for _, p in params:
        n = p.numel()
        if bits <= 0:
            total += 32 * n
        elif topk < 1.0:
            k = max(1, int(round(topk * n)))
            total += 32 + k * (bits + max(1, math.ceil(math.log2(n))))
        else:
            total += 32 + n * bits
    return total


def matched_period(codec: SemanticCodec, ncfg) -> int:
    """Period at which synchronising *all* receiver levels together costs the
    same bits per step as the nested schedule."""
    if ncfg["single_period"] != "auto":
        return int(ncfg["single_period"])
    groups = codec.groups()
    bits, topk = int(ncfg["sync"]["bits"]), float(ncfg["sync"]["topk"])
    b_adapter = _dense_bits(groups["rx_adapter"], bits, topk)
    b_lora = _dense_bits(groups["rx_lora"], bits, topk)
    nested_rate = b_adapter / int(ncfg["periods"]["medium"]) + b_lora / int(ncfg["periods"]["slow"])
    return max(1, int(round((b_adapter + b_lora) / nested_rate)))


def build_levels(codec: SemanticCodec, method: str, ncfg) -> list[Level]:
    if method not in METHODS:
        raise ValueError(f"unknown method '{method}'; choose from {METHODS}")
    groups = codec.groups()
    periods, lrs = ncfg["periods"], ncfg["lrs"]
    fast, medium, slow = int(periods["fast"]), int(periods["medium"]), int(periods["slow"])
    if method == "frozen":
        return []
    if method in ("single_rate", "ewc", "full_online"):
        period = 1 if method == "full_online" else matched_period(codec, ncfg)
        lr = float(ncfg["single_lr"])
        return [
            Level("tx_fast", "tx", groups["tx_fast"], period, lr),
            Level("tx_lora", "tx", groups["tx_lora"], period, lr),
            Level("rx_adapter", "rx", groups["rx_adapter"], period, lr),
            Level("rx_lora", "rx", groups["rx_lora"], period, lr),
        ]
    tx_levels = [
        Level("tx_fast", "tx", groups["tx_fast"], fast, float(lrs["fast"])),
        Level("tx_lora", "tx", groups["tx_lora"], medium, float(lrs["medium"])),
    ]
    if method == "tx_only":
        return tx_levels
    if method == "nested_two_level":
        period = matched_period(codec, ncfg)
        return tx_levels + [
            Level("rx_adapter", "rx", groups["rx_adapter"], period, float(lrs["medium"])),
            Level("rx_lora", "rx", groups["rx_lora"], period, float(lrs["slow"])),
        ]
    use_ema = method == "nested" and bool(ncfg["ema"]["enabled"])
    return tx_levels + [
        Level("rx_adapter", "rx", groups["rx_adapter"], medium, float(lrs["medium"])),
        Level("rx_lora", "rx", groups["rx_lora"], slow, float(lrs["slow"]), ema=use_ema),
    ]


def estimate_fisher(codec: SemanticCodec, params, texts: list[str], m: int, snr_db: float, n_batches: int, batch_size: int, seed: int):
    """Diagonal empirical Fisher of the base task (for the EWC baseline),
    normalised to unit mean so that ``ewc_lambda`` has a scale-free meaning."""
    rng = np.random.default_rng(seed)
    channel = Channel("awgn", seed=seed + 7)
    fisher = [torch.zeros_like(p) for p in params]
    for _ in range(n_batches):
        idx = rng.integers(0, len(texts), size=batch_size)
        batch = codec.tokenize([texts[int(i)] for i in idx])
        ce, _ = codec.loss_at(codec.encode(batch), batch, m, snr_db, channel)
        grads = torch.autograd.grad(ce.mean(), params, allow_unused=True)
        for f, g in zip(fisher, grads):
            if g is not None:
                f.add_(g.detach() ** 2 / n_batches)
    mean = sum(float(f.sum()) for f in fisher) / max(sum(f.numel() for f in fisher), 1)
    if mean > 0:
        for f in fisher:
            f.div_(mean)
    return fisher


def run_stream(cfg, method: str, seed: int, ckpt_dir: str | Path, out_path: str | Path) -> dict:
    set_seed(seed)
    device = get_device(cfg["device"])
    scfg, ncfg = cfg["stream"], cfg["nested"]
    codec = load_codec(cfg, ckpt_dir, device)
    codec.freeze_all()
    codec.eval()

    names = list(scfg["domains"]) if scfg["kind"] == "hier" else [scfg["temporal_corpus"]]
    corpora = {name: get_corpus(name, cfg["data"]) for name in names}
    stream = build_stream(corpora, scfg, seed)
    probes = build_probes(corpora, scfg, seed=0)  # identical probes for every method and seed
    sim = load_similarity(cfg["eval"]["sim_model"], device)

    levels = build_levels(codec, method, ncfg)
    for level in levels:
        for _, param in level.params:
            param.requires_grad_(True)
    optimizer = NestedOptimizer(
        levels,
        sync_bits=int(ncfg["sync"]["bits"]),
        sync_topk=float(ncfg["sync"]["topk"]),
        ema_decay=float(ncfg["ema"]["decay"]),
        ema_beta=float(ncfg["ema"]["beta"]),
        grad_clip=float(ncfg["grad_clip"]),
    )
    trainable = optimizer.trainable()

    m_op = int(scfg["m_op"])
    snr_mean = float(scfg["snr"]["mean"])
    channel = Channel(cfg["channel"]["type"], seed=seed * 1000 + 1)
    probe_channel = Channel(cfg["channel"]["type"], seed=777)
    rng = np.random.default_rng(seed + 99)

    fisher, anchor = None, None
    if method == "ewc" and trainable:
        base_texts = [t for name in cfg["data"]["base"] for t in get_corpus(name, cfg["data"]).texts("train")]
        fisher = estimate_fisher(codec, trainable, base_texts, m_op, snr_mean, int(ncfg["ewc_batches"]), int(scfg["batch_size"]), seed)
        anchor = [p.detach().clone() for p in trainable]

    def probe(step: int, regime: int) -> dict:
        row = {"step": step, "regime": regime, "domains": {}}
        for name, texts in probes.items():
            probe_channel.reseed(777)
            entry = evaluate_tf(codec, texts, m_op, snr_mean, probe_channel, batch_size=int(cfg["eval"]["batch_size"]))
            probe_channel.reseed(778)
            subset = texts[: int(scfg["gen_sentences"]) * 4]
            entry.update(evaluate_gen(codec, subset, m_op, snr_mean, probe_channel, sim, batch_size=int(cfg["eval"]["batch_size"])))
            row["domains"][name] = entry
        return row

    timer = Timer()
    log = {k: [] for k in ("ce_bits", "tok_acc", "snr_db", "domain", "topic", "regime")}
    gen_log, probe_log = [], [probe(0, -1)]
    n_steps = len(stream)
    print(f"[stream] method={method} seed={seed} steps={n_steps} device={device} levels=" + ", ".join(f"{lv.name}@{lv.period}" for lv in levels))

    for item in stream:
        batch = codec.tokenize(item.texts)
        if int(scfg["gen_every"]) > 0 and item.t % int(scfg["gen_every"]) == 0:
            subset = item.texts[: int(scfg["gen_sentences"])]
            gen = evaluate_gen(codec, subset, m_op, item.snr_db, channel, sim)
            gen_log.append({"step": item.t, "domain": item.domain, **gen})

        if not levels:
            with torch.no_grad():
                ce, acc = codec.loss_at(codec.encode(batch), batch, m_op, item.snr_db, channel)
        else:
            z = codec.encode(batch)
            ce, acc = codec.loss_at(z, batch, m_op, item.snr_db, channel)  # scored before any update
            loss = ce.mean()
            extra = sample_grans([m for m in codec.grans if m != m_op], int(scfg["extra_grans"]), rng)
            for m in extra:  # keep the other operating points alive (rate nesting)
                loss = loss + codec.loss_at(z, batch, m, item.snr_db, channel)[0].mean()
            loss = loss / (1 + len(extra))
            if fisher is not None:
                penalty = sum((f * (p - a) ** 2).sum() for f, p, a in zip(fisher, trainable, anchor))
                loss = loss + 0.5 * float(ncfg["ewc_lambda"]) * penalty
            loss.backward()
            optimizer.step(item.t)

        log["ce_bits"].append(ce.detach().mean().item() / LN2)
        log["tok_acc"].append(acc.detach().mean().item())
        log["snr_db"].append(item.snr_db)
        log["domain"].append(item.domain)
        log["topic"].append(item.topic)
        log["regime"].append(item.regime)
        if item.regime_end:
            probe_log.append(probe(item.t, item.regime))
        if item.t % 50 == 0 or item.t == n_steps:
            recent = float(np.mean(log["tok_acc"][-50:]))
            print(f"[stream] {method} step {item.t:>5}/{n_steps} domain={item.domain:<10} tok_acc(50)={recent:.3f} sync_bits={optimizer.ledger.total_bits:,} it/s={item.t / max(timer.elapsed(), 1e-9):.2f}")

    final = probe_log[-1]["domains"]
    sync_bits = optimizer.ledger.total_bits
    summary = {
        "mean_tok_acc": float(np.mean(log["tok_acc"])),
        "mean_ce_bits": float(np.mean(log["ce_bits"])),
        "final_probe_tok_acc": float(np.mean([v["tok_acc"] for v in final.values()])),
        "final_probe_bleu1": float(np.mean([v["bleu1"] for v in final.values()])),
        "final_probe_bleu4": float(np.mean([v["bleu4"] for v in final.values()])),
        "sync_bits": sync_bits,
        "sync_bits_per_step": sync_bits / n_steps,
        "net_rate": net_semantic_rate(m_op, sync_bits, n_steps, float(cfg["link"]["traffic_per_step"]), snr_mean),
        "seconds": timer.elapsed(),
    }
    result = {
        "method": method,
        "seed": seed,
        "stream": scfg["kind"],
        "m_op": m_op,
        "n_steps": n_steps,
        "levels": [
            {"name": lv.name, "side": lv.side, "period": lv.period, "lr": lv.lr, "ema": lv.ema, "n_params": lv.n_params, "bits_per_update": optimizer.expected_bits_per_update(lv)}
            for lv in levels
        ],
        "summary": summary,
        "log": log,
        "gen": gen_log,
        "probes": probe_log,
        "ledger": {**optimizer.ledger.summary(), "cumulative": optimizer.ledger.cumulative(n_steps)},
        "config": cfg.to_dict(),
        "env": environment(),
    }
    save_json(out_path, result)
    print(f"[stream] {method} seed={seed}: mean tok_acc={summary['mean_tok_acc']:.4f} final-probe tok_acc={summary['final_probe_tok_acc']:.4f} sync={sync_bits:,} bits -> {out_path}")
    return result
