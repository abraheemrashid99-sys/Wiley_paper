"""Uncertainty-triggered incremental refinement.

Because every shell of the payload is power-normalised on its own, the
receiver can start from the shortest prefix and ask for the next shell only
when it is not confident in what it decoded: semantic incremental redundancy
without re-encoding. The receiver's confidence is *calibrated* on held-out
data so that the stopping threshold has a meaning (a target probability that
the reconstruction is acceptable), and the calibration is itself reported.

Every received vector is drawn once per message; shells are then revealed
progressively, so a longer prefix literally contains the shorter one.
"""

from __future__ import annotations

import numpy as np
import torch

from .channel import Channel
from .codec import SemanticCodec
from .metrics import expected_calibration_error, sentence_bleu, words


def _quality(kind: str, references: list[str], hypotheses: list[str], sim=None) -> np.ndarray:
    if kind == "sim":
        if sim is None:
            raise RuntimeError("refine.quality='sim' needs eval.sim_model to be available")
        return np.asarray(sim(references, hypotheses), dtype=float)
    if kind == "exact":
        return np.array([float(words(r) == words(h)) for r, h in zip(references, hypotheses)])
    n = {"bleu1": 1, "bleu2": 2, "bleu3": 3, "bleu4": 4}[kind]
    return np.array([sentence_bleu(r, h, n) for r, h in zip(references, hypotheses)])


@torch.no_grad()
def collect(codec: SemanticCodec, texts: list[str], snr_db: float, channel: Channel, quality: str, sim=None, batch_size: int = 32) -> dict:
    """Decode every message at every granularity from one channel realisation.

    Returns arrays of shape (N, K): ``quality`` and the three confidence
    statistics of the greedy decode."""
    quality_rows, lp_rows, ent_rows, minp_rows = [], [], [], []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        batch = codec.tokenize(chunk)
        y_full, _ = channel(codec.encode(batch), snr_db)
        q, lp, ent, minp = [], [], [], []
        for m in codec.grans:
            decoded = codec.generate(codec.prompts(codec.mask_prefix(y_full, m), m), max_new=int(batch.ids.shape[1]) + 4)
            q.append(_quality(quality, chunk, decoded.texts, sim))
            lp.append(decoded.mean_logprob.numpy())
            ent.append(decoded.mean_entropy.numpy())
            minp.append(decoded.min_prob.numpy())
        quality_rows.append(np.stack(q, 1))
        lp_rows.append(np.stack(lp, 1))
        ent_rows.append(np.stack(ent, 1))
        minp_rows.append(np.stack(minp, 1))
    return {
        "quality": np.concatenate(quality_rows),
        "mean_logprob": np.concatenate(lp_rows),
        "mean_entropy": np.concatenate(ent_rows),
        "min_prob": np.concatenate(minp_rows),
    }


def _features(data: dict) -> np.ndarray:
    n, k = data["quality"].shape
    gran = np.tile(np.arange(k, dtype=float) / max(k - 1, 1), (n, 1))
    return np.stack([data["mean_logprob"], data["mean_entropy"], data["min_prob"], gran], axis=-1).reshape(n * k, 4)


class Calibrator:
    """Logistic (Platt-style) map from decode statistics to P(acceptable)."""

    def __init__(self, ridge: float = 1e-2):
        self.ridge = ridge
        self.mean = self.std = self.weights = None

    def fit(self, features: np.ndarray, labels: np.ndarray, iters: int = 200) -> "Calibrator":
        self.mean, self.std = features.mean(0), features.std(0) + 1e-8
        x = np.hstack([(features - self.mean) / self.std, np.ones((len(features), 1))])
        w = np.zeros(x.shape[1])
        for _ in range(iters):  # Newton / IRLS with a ridge penalty
            p = 1.0 / (1.0 + np.exp(-np.clip(x @ w, -30, 30)))
            grad = x.T @ (p - labels) / len(labels) + self.ridge * w
            hess = (x * (p * (1 - p))[:, None]).T @ x / len(labels) + self.ridge * np.eye(len(w))
            step = np.linalg.solve(hess, grad)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        self.weights = w
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = np.hstack([(features - self.mean) / self.std, np.ones((len(features), 1))])
        return 1.0 / (1.0 + np.exp(-np.clip(x @ self.weights, -30, 30)))


def stopping_curve(prob: np.ndarray, quality: np.ndarray, grans: list[int], thresholds: np.ndarray, gamma: float) -> list[dict]:
    """Stop at the first shell whose calibrated confidence reaches the
    threshold; otherwise use everything."""
    n, k = prob.shape
    sizes = np.asarray(grans, dtype=float)
    rows = []
    for tau in thresholds:
        hit = prob >= tau
        stop = np.where(hit.any(1), hit.argmax(1), k - 1)
        q = quality[np.arange(n), stop]
        rows.append({"threshold": float(tau), "symbols": float(sizes[stop].mean()), "quality": float(q.mean()), "success": float((q >= gamma).mean()), "rounds": float((stop + 1).mean())})
    return rows


def reference_policies(prob: np.ndarray, quality: np.ndarray, grans: list[int], gamma: float, seed: int = 0) -> dict:
    n, k = quality.shape
    sizes = np.asarray(grans, dtype=float)
    fixed = [{"m": int(grans[j]), "symbols": float(sizes[j]), "quality": float(quality[:, j].mean()), "success": float((quality[:, j] >= gamma).mean())} for j in range(k)]
    ok = quality >= gamma
    stop = np.where(ok.any(1), ok.argmax(1), k - 1)
    q = quality[np.arange(n), stop]
    oracle = {"symbols": float(sizes[stop].mean()), "quality": float(q.mean()), "success": float((q >= gamma).mean())}
    return {"fixed": fixed, "oracle": oracle}


def random_matched(stop: np.ndarray, quality: np.ndarray, grans: list[int], gamma: float, seed: int = 0) -> dict:
    """Same distribution of stopping points, assigned to messages at random:
    the control that shows whether the confidence signal carries information."""
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(stop)
    q = quality[np.arange(len(stop)), shuffled]
    return {"symbols": float(np.asarray(grans, dtype=float)[shuffled].mean()), "quality": float(q.mean()), "success": float((q >= gamma).mean())}


def run_refinement(codec: SemanticCodec, val_texts: list[str], test_texts: list[str], snr_db: float, channel_kind: str, rcfg, sim=None, batch_size: int = 32) -> dict:
    gamma, kind = float(rcfg["gamma"]), rcfg["quality"]
    val = collect(codec, val_texts, snr_db, Channel(channel_kind, seed=4101), kind, sim, batch_size)
    test = collect(codec, test_texts, snr_db, Channel(channel_kind, seed=4102), kind, sim, batch_size)
    val_labels = (val["quality"] >= gamma).astype(float).reshape(-1)
    test_labels = (test["quality"] >= gamma).astype(float).reshape(-1)

    calibrator = Calibrator().fit(_features(val), val_labels)
    prob = calibrator.predict(_features(test)).reshape(test["quality"].shape)
    raw = np.exp(test["mean_logprob"]).reshape(-1)  # uncalibrated confidence
    # Sweep the whole operating curve, from "always stop at the first shell" to
    # "never stop early". Thresholds are quantiles of the predicted
    # probabilities themselves (no labels involved).
    thresholds = np.unique(np.concatenate([np.quantile(prob.reshape(-1), np.linspace(0.0, 1.0, 21)), [np.inf]]))
    curve = stopping_curve(prob, test["quality"], codec.grans, thresholds, gamma)

    n, k = prob.shape
    mid = thresholds[(len(thresholds) - 1) // 2]
    hit = prob >= mid
    stop = np.where(hit.any(1), hit.argmax(1), k - 1)
    return {
        "snr_db": snr_db,
        "channel": channel_kind,
        "quality_metric": kind,
        "gamma": gamma,
        "n_val": len(val_texts),
        "n_test": len(test_texts),
        "base_rate": float(test_labels.mean()),
        "ece_raw": expected_calibration_error(raw, test_labels),
        "ece_calibrated": expected_calibration_error(prob.reshape(-1), test_labels),
        "curve": curve,
        "random_matched_at_mid_threshold": {"threshold": float(mid), **random_matched(stop, test["quality"], codec.grans, gamma)},
        **reference_policies(prob, test["quality"], codec.grans, gamma),
    }
