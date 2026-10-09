"""Base-codec pretraining and the shared evaluation routines."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from .channel import Channel
from .codec import SemanticCodec, build_codec, save_codec
from .data import get_corpus
from .metrics import fidelity
from .utils import JsonlLogger, Timer, count_params, ensure_dir, environment, get_device, save_json, set_seed

LN2 = math.log(2.0)


def sample_grans(grans: list[int], k: int, rng: np.random.Generator, always: int | None = None) -> list[int]:
    """Stochastic nesting: ``k`` granularities per step, drawn without
    replacement. Averaged over steps this is an unbiased estimate of the
    uniform multi-granularity (Matryoshka) objective at ``k/|M|`` of its cost."""
    pool = [m for m in grans if m != always]
    k_free = max(0, min(k - (always is not None), len(pool)))
    picked = [int(m) for m in rng.choice(pool, size=k_free, replace=False)] if k_free else []
    return ([int(always)] if always is not None else []) + picked


def lr_at(step: int, total: int, warmup: int) -> float:
    """Linear warm-up followed by cosine annealing (multiplier in [0, 1])."""
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


@torch.no_grad()
def evaluate_tf(codec: SemanticCodec, texts: list[str], m: int, snr_db, channel: Channel, batch_size: int = 32) -> dict:
    """Teacher-forced metrics: cross-entropy (bits/token) and token accuracy."""
    ce_all, acc_all = [], []
    for i in range(0, len(texts), batch_size):
        batch = codec.tokenize(texts[i : i + batch_size])
        ce, acc = codec.loss_at(codec.encode(batch), batch, m, snr_db, channel)
        ce_all.append(ce.float().cpu())
        acc_all.append(acc.float().cpu())
    ce, acc = torch.cat(ce_all), torch.cat(acc_all)
    return {"ce_bits": float(ce.mean()) / LN2, "tok_acc": float(acc.mean()), "n": len(texts)}


@torch.no_grad()
def evaluate_gen(codec: SemanticCodec, texts: list[str], m: int, snr_db, channel: Channel, sim=None, batch_size: int = 32, return_texts: bool = False) -> dict:
    """Free-running (greedy) reconstruction metrics: BLEU-1..4, sacreBLEU,
    exact match and, when a similarity model is supplied, sentence similarity."""
    hyps: list[str] = []
    for i in range(0, len(texts), batch_size):
        hyps.extend(codec.reconstruct(texts[i : i + batch_size], m, snr_db, channel).texts)
    out = fidelity(texts, hyps, sim)
    out["n"] = len(texts)
    if return_texts:
        out["hyps"] = hyps
    return out


def pretrain(cfg) -> Path:
    """Train the base codec on the base corpora and save it to
    ``<out_dir>/base``. Returns the checkpoint directory."""
    set_seed(int(cfg["seed"]))
    device = get_device(cfg["device"])
    out_dir = ensure_dir(Path(cfg["out_dir"]) / "base")
    pcfg = cfg["pretrain"]

    train_texts, val_texts = [], []
    for name in cfg["data"]["base"]:
        corpus = get_corpus(name, cfg["data"])
        train_texts += corpus.texts("train")
        val_texts += corpus.texts("val")
    rng = np.random.default_rng(int(cfg["seed"]))
    val_texts = [val_texts[int(i)] for i in rng.permutation(len(val_texts))[: int(pcfg["eval_sentences"])]]

    codec = build_codec(cfg, device)
    groups = codec.groups()
    head_params = [p for g in ("tx_fast", "rx_adapter", "rx_static") for _, p in groups[g]]
    lora_params = [p for g in ("tx_lora", "rx_lora") for _, p in groups[g]]
    param_groups = [
        {"params": head_params, "lr": float(pcfg["lr"]), "base_lr": float(pcfg["lr"])},
        {"params": lora_params, "lr": float(pcfg["lora_lr"]), "base_lr": float(pcfg["lora_lr"])},
    ]
    if codec.train_backbone:
        seen = {id(p) for p in head_params + lora_params}
        backbone = [p for p in codec.parameters() if p.requires_grad and id(p) not in seen]
        param_groups.append({"params": backbone, "lr": float(pcfg["lora_lr"]), "base_lr": float(pcfg["lora_lr"])})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=float(pcfg["weight_decay"]))
    trainable = [p for group in param_groups for p in group["params"]]

    channel = Channel(cfg["channel"]["type"], seed=int(cfg["seed"]) * 1000 + 1)
    eval_channel = Channel(cfg["channel"]["type"], seed=12345)
    snr_lo, snr_hi = cfg["channel"]["snr_train"]
    steps, batch_size = int(pcfg["steps"]), int(pcfg["batch_size"])
    logger = JsonlLogger(out_dir / "pretrain_log.jsonl")
    timer = Timer()
    print(
        f"[pretrain] device={device} backbone={cfg['backbone']['name']} "
        f"trainable={count_params(trainable):,} train_sentences={len(train_texts):,} steps={steps}"
    )

    history = []
    codec.train()
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * lr_at(step, steps, int(pcfg["warmup"]))
        idx = rng.integers(0, len(train_texts), size=batch_size)
        batch = codec.tokenize([train_texts[int(i)] for i in idx])
        snr = float(rng.uniform(snr_lo, snr_hi))
        z = codec.encode(batch)
        grans = sample_grans(codec.grans, int(pcfg["grans_per_step"]), rng)
        loss, acc_log = 0.0, {}
        for m in grans:
            ce, acc = codec.loss_at(z, batch, m, snr, channel)
            loss = loss + ce.mean() / len(grans)
            acc_log[m] = acc.detach().mean().item()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, float(pcfg["grad_clip"]))
        optimizer.step()

        if (step + 1) % int(pcfg["log_every"]) == 0 or step == 0:
            rate = (step + 1) / max(timer.elapsed(), 1e-9)
            print(f"[pretrain] step {step + 1:>6}/{steps} loss={loss.item():.4f} bits/token={loss.item() / LN2:.3f} it/s={rate:.2f}")
            logger.log(step=step + 1, loss=loss.item(), snr=snr, acc=acc_log, it_per_s=rate)
        if (step + 1) % int(pcfg["eval_every"]) == 0 or step + 1 == steps:
            codec.eval()
            eval_channel.reseed(12345)
            row = {"step": step + 1}
            for m in codec.grans:
                row[str(m)] = evaluate_tf(codec, val_texts, m, 10.0, eval_channel, batch_size=batch_size)
            history.append(row)
            logger.log(eval=row)
            print("[pretrain] val tok_acc @10dB: " + "  ".join(f"m={m}:{row[str(m)]['tok_acc']:.3f}" for m in codec.grans))
            codec.train()

    codec.eval()
    extra = {"steps": steps, "seconds": timer.elapsed(), "history": history}
    save_codec(codec, out_dir, extra)
    save_json(out_dir / "pretrain_summary.json", {"config": cfg.to_dict(), "env": environment(), **extra})
    logger.close()
    print(f"[pretrain] saved {out_dir / 'codec.pt'} in {timer.elapsed() / 60:.1f} min")
    return out_dir
