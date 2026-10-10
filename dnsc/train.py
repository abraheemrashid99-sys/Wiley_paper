"""Codebook pretraining and the shared evaluation routines.

The only component that is trained before deployment is the semantic
codebook. Its objective is a channel-decoding loss - recover the transmitted
token from its noisy codeword prefix - so no LLM forward or backward pass is
needed and the whole pretraining takes minutes on a CPU.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .channel import Channel
from .codec import SemanticCodec, TokenCache, build_codec, save_codec
from .data import get_corpus
from .metrics import fidelity
from .utils import JsonlLogger, Timer, count_params, ensure_dir, environment, get_device, save_json, set_seed

LN2 = math.log(2.0)


def sample_grans(grans: list[int], k: int, rng: np.random.Generator, always: int | None = None) -> list[int]:
    """``k`` granularities drawn without replacement, optionally always
    including one of them (the operating point)."""
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


def flat_tokens(codec: SemanticCodec, documents: list[list[str]], chunk: int = 2000) -> np.ndarray:
    """All token ids of the documents as one 1-D array, tokenised as the
    codec tokenises a stream: the sentence that opens a document on its own,
    every later one as running text (truncated like the codec)."""
    texts = [(" " + text) if i else text for doc in documents for i, text in enumerate(doc)]
    out: list[int] = []
    for i in range(0, len(texts), chunk):
        for ids in codec.tok(texts[i : i + chunk], add_special_tokens=False)["input_ids"]:
            out.extend(ids[: codec.max_tokens])
    return np.asarray(out, dtype=np.int64)


def unigram_log_probs(ids: np.ndarray, vocab: int) -> torch.Tensor:
    """Smoothed unigram log-probabilities (add one half per type)."""
    counts = np.bincount(ids, minlength=vocab).astype(np.float64) + 0.5
    return torch.tensor(np.log(counts / counts.sum()), dtype=torch.float32)


def codebook_loss(codec: SemanticCodec, ids: torch.Tensor, snr_db: torch.Tensor, channel: Channel, grans: list[int]) -> tuple[torch.Tensor, dict[int, float]]:
    """Mean cross-entropy of channel-only decoding over ``grans`` (a subset of
    the codec's granularities).

    The batch of tokens is one long block, so block power control is the
    identity up to sampling noise and is omitted; every token sees its own
    noise level (and, under fading, its own gain)."""
    code = codec.codebook()
    x = code[ids]  # (N, M)
    eff = channel.effective_snr_db(len(ids), snr_db)
    noise = channel.noise(x.shape, torch.pow(10.0, -eff / 20.0).unsqueeze(1))
    y = x + noise.to(device=x.device, dtype=x.dtype)
    prec = torch.pow(10.0, eff / 10.0).to(device=x.device, dtype=x.dtype).unsqueeze(1)  # (N, 1)
    loss, acc = 0.0, {}
    for m in grans:
        prefix = code[:, :m]
        # log-likelihood up to a codeword-independent term: w (y.c - |c|^2 / 2)
        logits = prec * torch.addmm(-0.5 * (prefix * prefix).sum(-1).unsqueeze(0), y[:, :m], prefix.t())
        loss = loss + F.cross_entropy(logits, ids) / len(grans)
        acc[m] = (logits.detach().argmax(-1) == ids).float().mean().item()
    return loss, acc


@torch.no_grad()
def codebook_accuracy(codec: SemanticCodec, ids: torch.Tensor, snr_db: float, channel: Channel, chunk: int = 256) -> dict[int, float]:
    """Channel-only token accuracy at every granularity, in memory-safe chunks."""
    hits = {m: 0.0 for m in codec.grans}
    for i in range(0, len(ids), chunk):
        part = ids[i : i + chunk]
        _, acc = codebook_loss(codec, part, torch.full((len(part),), float(snr_db)), channel, codec.grans)
        for m in codec.grans:
            hits[m] += acc[m] * len(part)
    return {m: hits[m] / max(len(ids), 1) for m in codec.grans}


def prior_bits(log_prior: torch.Tensor, batch) -> torch.Tensor:
    """Per-example cross-entropy (bits/token) of the prior alone: what the
    receiver knows about the text before a single symbol arrives."""
    valid = batch.mask.to(log_prior.dtype)
    nll = -log_prior.gather(-1, batch.ids.unsqueeze(-1)).squeeze(-1)
    return (nll * valid).sum(1) / batch.lens.to(log_prior.dtype) / LN2


def _slice(values, start: int, stop: int):
    return None if values is None else values[start:stop]


@torch.no_grad()
def evaluate_tf(
    codec: SemanticCodec,
    texts: list[str],
    m: int,
    snr_db,
    channel: Channel,
    prior: str = "llm",
    cache: TokenCache | None = None,
    batch_size: int = 16,
    contexts: list[list[int]] | None = None,
    continued: list[bool] | None = None,
) -> dict:
    """Teacher-forced posterior metrics: cross-entropy (bits/token) and token
    accuracy, averaged over sentences, plus the cross-entropy of the prior
    alone. ``contexts`` / ``continued`` give, per text, the token ids that
    precede it and whether it continues a document."""
    ce_all, acc_all, prior_all = [], [], []
    for i in range(0, len(texts), batch_size):
        batch = codec.tokenize(texts[i : i + batch_size], _slice(continued, i, i + batch_size))
        y, prec = codec.transmit(codec.encode(batch), batch.lens, snr_db, channel)
        log_prior = codec.prior_tf(batch, cache, prior, _slice(contexts, i, i + batch_size))
        ce, acc, _ = codec.loss_at(batch, y, prec, m, log_prior)
        ce_all.append(ce.float().cpu())
        acc_all.append(acc.float().cpu())
        if prior != "none":
            prior_all.append(prior_bits(log_prior, batch).float().cpu())
    ce, acc = torch.cat(ce_all), torch.cat(acc_all)
    out = {"ce_bits": float(ce.mean()) / LN2, "tok_acc": float(acc.mean()), "n": len(texts)}
    if prior_all:
        out["prior_bits"] = float(torch.cat(prior_all).mean())
    return out


@torch.no_grad()
def evaluate_gen(
    codec: SemanticCodec,
    texts: list[str],
    m: int,
    snr_db,
    channel: Channel,
    sim=None,
    prior: str = "llm",
    cache: TokenCache | None = None,
    batch_size: int = 16,
    return_texts: bool = False,
    contexts: list[list[int]] | None = None,
    continued: list[bool] | None = None,
) -> dict:
    """Free-running (greedy MAP) reconstruction metrics: token accuracy,
    BLEU-1..4, sacreBLEU, exact match and, with a similarity model, sentence
    similarity. References are the texts as a perfect link would return them."""
    refs: list[str] = []
    hyps: list[str] = []
    correct = total = 0
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        flags = _slice(continued, i, i + batch_size)
        batch = codec.tokenize(chunk, flags)
        y, prec = codec.transmit(codec.encode(batch), batch.lens, snr_db, channel)
        decoded = codec.decode(y, m, prec, batch.lens, prior, cache, _slice(contexts, i, i + batch_size))
        refs.extend(codec.reference_texts(chunk, flags))
        hyps.extend(decoded.texts)
        for row, length, got in zip(batch.ids.tolist(), batch.lens.tolist(), decoded.token_ids):
            correct += sum(int(a == b) for a, b in zip(row[:length], got))
            total += length
    out = fidelity(refs, hyps, sim)
    out["gen_tok_acc"] = correct / max(total, 1)
    out["n"] = len(texts)
    if return_texts:
        out["refs"], out["hyps"] = refs, hyps
    return out


def pretrain(cfg, force: bool = False) -> Path:
    """Train the semantic codebook on the base corpora and save the codec to
    ``<out_dir>/base``. A finished checkpoint is left alone unless ``force``."""
    set_seed(int(cfg["seed"]))
    device = get_device(cfg["device"])
    out_dir = ensure_dir(Path(cfg["out_dir"]) / "base")
    pcfg = cfg["pretrain"]
    if force:
        (out_dir / "codec.pt").unlink(missing_ok=True)
    if (out_dir / "codec.pt").exists():
        print(f"[pretrain] {out_dir / 'codec.pt'} already exists - nothing to do (use --force to retrain)")
        return out_dir

    train_texts, val_texts = [], []
    for name in cfg["data"]["base"]:
        corpus = get_corpus(name, cfg["data"])
        train_texts += [[r["text"] for r in doc] for doc in corpus.documents("train")]
        val_texts += [[r["text"] for r in doc] for doc in corpus.documents("val")]

    codec = build_codec(cfg, device)
    train_ids = flat_tokens(codec, train_texts)
    val_flat = flat_tokens(codec, val_texts)
    codec.log_unigram.copy_(unigram_log_probs(train_ids, codec.vocab).to(device))
    rng = np.random.default_rng(int(cfg["seed"]))
    val_ids = torch.tensor(val_flat[rng.permutation(len(val_flat))[: int(pcfg["eval_tokens"])]], device=device)

    params = [p for _, p in codec.groups()["codebook"]]
    optimizer = torch.optim.AdamW(params, lr=float(pcfg["lr"]), weight_decay=float(pcfg["weight_decay"]))
    channel = Channel(cfg["channel"]["type"], seed=int(cfg["seed"]) * 1000 + 1)
    eval_channel = Channel("awgn", seed=12345)
    snr_lo, snr_hi = cfg["channel"]["snr_train"]
    steps, n_tokens = int(pcfg["steps"]), int(pcfg["batch_tokens"])
    n_uniform = int(round(float(pcfg["uniform_share"]) * n_tokens))
    logger = JsonlLogger(out_dir / "pretrain_log.jsonl")
    timer = Timer()
    print(
        f"[pretrain] device={device} threads={torch.get_num_threads()} backbone={cfg['backbone']['name']} vocab={codec.vocab:,} "
        f"codebook_params={count_params(params):,} train_tokens={len(train_ids):,} steps={steps}"
    )

    history = []
    codec.train_codebook = True
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = float(pcfg["lr"]) * lr_at(step, steps, int(pcfg["warmup"]))
        picked = train_ids[rng.integers(0, len(train_ids), size=n_tokens)]
        if n_uniform:  # every token of the vocabulary must be decodable, not only those of the base corpus
            picked[:n_uniform] = rng.integers(0, codec.vocab, size=n_uniform)
        ids = torch.tensor(picked, device=device)
        snr = torch.tensor(rng.uniform(snr_lo, snr_hi, size=n_tokens), dtype=torch.float32)
        loss, acc = codebook_loss(codec, ids, snr, channel, sample_grans(codec.grans, int(pcfg["grans_per_step"]), rng))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, float(pcfg["grad_clip"]))
        optimizer.step()
        codec.invalidate_codebook()

        if (step + 1) % int(pcfg["log_every"]) == 0 or step == 0:
            rate = (step + 1) / max(timer.elapsed(), 1e-9)
            eta_min = (steps - step - 1) / max(rate, 1e-9) / 60
            print(f"[pretrain] step {step + 1:>5}/{steps} loss={loss.item():.4f} it/s={rate:.2f} eta={eta_min:.1f}min")
            logger.log(step=step + 1, loss=loss.item(), acc=acc, it_per_s=rate)
        if (step + 1) % int(pcfg["eval_every"]) == 0 or step + 1 == steps:
            eval_channel.reseed(12345)
            val_acc = codebook_accuracy(codec, val_ids, 10.0, eval_channel)
            row = {"step": step + 1, **{str(m): val_acc[m] for m in codec.grans}}
            history.append(row)
            logger.log(eval=row)
            print("[pretrain] channel-only token accuracy @10dB (no prior): " + "  ".join(f"m={m}:{val_acc[m]:.3f}" for m in codec.grans))

    codec.train_codebook = False
    codec.invalidate_codebook()
    codec.eval()
    extra = {"steps": steps, "seconds": timer.elapsed(), "history": history}
    save_codec(codec, out_dir, extra)
    save_json(out_dir / "pretrain_summary.json", {"config": cfg.to_dict(), "env": environment(), **extra})
    logger.close()
    print(f"[pretrain] saved {out_dir / 'codec.pt'} in {timer.elapsed() / 60:.1f} min")
    return out_dir
