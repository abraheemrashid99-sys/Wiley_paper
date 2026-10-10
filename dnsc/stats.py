"""Statistical procedures used for every comparison in the paper.

Paired tests throughout (methods are evaluated on the same seeds or the same
test sentences), Holm correction across families, and an explicit equivalence
procedure for "no meaningful difference" claims.
"""

from __future__ import annotations

import itertools

import numpy as np


def mean_sd(values) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if values.size > 1 else 0.0


def paired_bootstrap(a, b, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Percentile CI of mean(a - b) with a shared resampling index."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired samples must have the same shape")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    diffs = (a[idx] - b[idx]).mean(axis=1)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {"delta": float((a - b).mean()), "ci_low": float(lo), "ci_high": float(hi)}


def paired_permutation(a, b, n_perm: int = 10000, seed: int = 0) -> float:
    """Two-sided sign-flip permutation p-value for mean(a - b) = 0.

    Enumerates all sign patterns exactly when there are at most 12 pairs
    (the usual case of one value per seed)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    observed = abs(d.mean())
    n = len(d)
    if n <= 12:
        signs = np.array(list(itertools.product([1.0, -1.0], repeat=n)))
        stats = np.abs((signs * d).mean(axis=1))
        return float((stats >= observed - 1e-12).mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([1.0, -1.0], size=(n_perm, n))
    stats = np.abs((signs * d).mean(axis=1))
    return float(((stats >= observed - 1e-12).sum() + 1) / (n_perm + 1))


def holm(pvalues) -> list[float]:
    """Holm step-down adjusted p-values, returned in the input order."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def equivalence(a, b, margin: float, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Two-one-sided-tests decision: equivalent iff the (1 - alpha) bootstrap
    interval of the paired difference lies strictly inside (-margin, +margin).
    The margin must be fixed before the results are inspected."""
    out = paired_bootstrap(a, b, n_boot=n_boot, alpha=alpha, seed=seed)
    out["margin"] = margin
    out["equivalent"] = bool(-margin < out["ci_low"] and out["ci_high"] < margin)
    return out


def block_bootstrap(series, block: int = 25, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Percentile CI of the mean of per-step paired differences.

    ``series`` is one sequence of differences per run (the same stream scored
    by two methods, so the channel noise and the messages cancel). Consecutive
    steps are correlated, so whole blocks of ``block`` steps, pooled over the
    runs, are resampled rather than single steps."""
    blocks = []
    for run in series:
        run = np.asarray(run, dtype=float)
        usable = len(run) - len(run) % block
        if usable:
            blocks.append(run[:usable].reshape(-1, block).mean(axis=1))
    if not blocks:
        raise ValueError("every run is shorter than one block")
    means = np.concatenate(blocks)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(means), size=(n_boot, len(means)))
    boot = means[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {"delta": float(means.mean()), "ci_low": float(lo), "ci_high": float(hi), "n_blocks": int(len(means)), "excludes_zero": bool(lo > 0 or hi < 0)}
