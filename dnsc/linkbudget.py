"""Net semantic rate and the adaptation crossover (derived quantities).

Keeping a learned codec aligned across a drifting link is not free: every
receiver-side update must be carried over the same air interface as the data.
The *net semantic rate* charges that cost to the messages it serves:

    R_net = m + B_sync / (C(snr) * N_msg)        [real symbols per message]

where ``m`` is the payload prefix length, ``B_sync`` the synchronisation bits
spent over a window, ``C(snr)`` the capacity per real symbol at which those
bits are assumed to travel (an optimistic, capacity-achieving assumption that
favours every adaptive method equally) and ``N_msg`` the number of messages
the link carried in that window.

The *crossover traffic* is the number of messages per adaptation step above
which adapting is cheaper than the alternative of not adapting and paying
``delta_m`` extra payload symbols per message to hold the same fidelity:

    N* = b_sync_per_step / (C(snr) * delta_m)

All inputs other than the measured ``B_sync`` and payload sizes are declared
parameters; outputs of this module are arithmetic, not measurements.
"""

from __future__ import annotations

import math

import numpy as np

from .channel import capacity_real


def sync_symbols(bits: float, snr_db: float) -> float:
    """Real channel uses needed to carry ``bits`` at capacity."""
    capacity = capacity_real(snr_db)
    return math.inf if capacity <= 0 else bits / capacity


def net_semantic_rate(m: float, sync_bits: float, n_steps: int, traffic_per_step: float, snr_db: float) -> float:
    messages = n_steps * traffic_per_step
    if messages <= 0:
        raise ValueError("the window must carry at least one message")
    return m + sync_symbols(sync_bits, snr_db) / messages


def crossover_traffic(sync_bits_per_step: float, delta_m: float, snr_db: float) -> float:
    """Messages per adaptation step above which adaptation pays for itself.

    ``delta_m`` is the payload saving per message at equal fidelity; when it
    is not positive, adaptation never pays and the crossover is infinite.
    """
    if delta_m <= 0:
        return math.inf
    return sync_symbols(sync_bits_per_step, snr_db) / delta_m


def overhead_fraction(m: float, sync_bits: float, n_steps: int, traffic_per_step: float, snr_db: float) -> float:
    """Share of the net rate that is synchronisation rather than payload."""
    net = net_semantic_rate(m, sync_bits, n_steps, traffic_per_step, snr_db)
    return (net - m) / net


def symbols_at_accuracy(symbols: list[float], accuracy: list[float], target: float) -> float:
    """Symbols per token needed to reach ``target`` accuracy, by linear
    interpolation along a measured operating curve. Returns infinity when the
    curve never reaches the target, and its cheapest point when every point
    already exceeds it."""
    points = sorted(zip(symbols, accuracy))
    if not points:
        return math.inf
    if points[0][1] >= target:
        return float(points[0][0])
    for (s0, a0), (s1, a1) in zip(points, points[1:]):
        if a1 >= target > a0:
            return float(s0 + (target - a0) / (a1 - a0) * (s1 - s0))
    return math.inf


def pooled_live(runs: list[dict], target: float) -> dict | None:
    """Symbols per token at ``target`` accuracy from the free-running audits
    of several runs (seeds) of one method, read off the operating curves
    averaged over the runs - every run audits the same number of steps, so
    this pools the audited messages. Reading the target off the pooled curve
    is more stable than averaging per-run readings, which are undefined
    whenever one run's few audited steps never reach the target.

    Returns ``None`` when a run has no audit; otherwise the fixed-prefix
    reference and the selective policy forward only and with its signalling
    (entropy and bitmap), the teacher-forced emulation of the fixed prefix
    and of the selective policy on exactly the audited steps, each ``None``
    if the pooled curve never reaches the target, and the number of audited
    steps."""
    curves = [r.get("live_curve") for r in runs]
    fixed = [r.get("live_fixed_curve") for r in runs]
    if not runs or not all(curves) or not all(fixed):
        return None

    def mean_curve(curves: list[list[dict]], key: str) -> list[float]:
        return [sum(c[i][key] for c in curves) / len(curves) for i in range(len(curves[0]))]

    def read(symbols: list[float], accuracy: list[float]) -> float | None:
        value = symbols_at_accuracy(symbols, accuracy, target)
        return value if math.isfinite(value) else None

    accuracy = mean_curve(curves, "tok_acc")
    grans = [float(p["m"]) for p in fixed[0]]
    emulated = all("emulated_tok_acc" in p for c in fixed for p in c) and all("emulated_tok_acc" in p for c in curves for p in c)
    return {
        "fixed": read(grans, mean_curve(fixed, "tok_acc")),
        "emulated_fixed": read(grans, mean_curve(fixed, "emulated_tok_acc")) if emulated else None,
        "emulated_forward": read(mean_curve(curves, "emulated_symbols"), mean_curve(curves, "emulated_tok_acc")) if emulated else None,
        "forward": read(mean_curve(curves, "symbols"), accuracy),
        "total_entropy": read(mean_curve(curves, "symbols_total_entropy"), accuracy),
        "total_bitmap": read(mean_curve(curves, "symbols_total"), accuracy),
        "audited_steps": sum(int(r["summary"].get("live_steps", 0)) for r in runs),
    }


def _audit_reading(steps: list[dict], grans: list[float], capacity: float, target: float, fallback_entropy: list[list[float]] | None = None, owners: list[int] | None = None) -> dict:
    """Read the free-running audit of a set of audited steps (``run["live"]``
    entries, possibly from several runs and with repetitions): the operating
    curves are averaged over the steps, the entropy of the requests is
    computed from their merged decisions, and the target is read off.

    Steps recorded before version 0.4 carry no decisions; then each step is
    charged the entropy rate of its run (``fallback_entropy[run][threshold]``,
    ``owners[i]`` = run of step ``i``)."""
    def binary_entropy(p: float) -> float:  # as dnsc.refine.binary_entropy, without importing torch here
        return 0.0 if p <= 0.0 or p >= 1.0 else -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)

    n_thresholds = len(steps[0]["live"])
    symbols = [sum(s["live"][j]["symbols"] for s in steps) / len(steps) for j in range(n_thresholds)]
    accuracy = [sum(s["live"][j]["tok_acc"] for s in steps) / len(steps) for j in range(n_thresholds)]
    bitmap = [sum(s["live"][j]["feedback_bits"] for s in steps) / len(steps) for j in range(n_thresholds)]
    entropy = []
    for j in range(n_thresholds):
        if all("meter" in s["live"][j] for s in steps):
            size = max(len(s["live"][j]["meter"]["open"]) for s in steps)
            opened, kept, tokens = [0.0] * size, [0.0] * size, 0.0
            for s in steps:
                m = s["live"][j]["meter"]
                for r, (o, c) in enumerate(zip(m["open"], m["cont"])):
                    opened[r] += o
                    kept[r] += c
                tokens += m["tokens"]
            entropy.append(sum(n * binary_entropy(c / n) for n, c in zip(opened, kept) if n > 0) / tokens if tokens else 0.0)
        else:
            entropy.append(sum(fallback_entropy[owners[i]][j] for i in range(len(steps))) / len(steps))
    fixed = [sum(s["live_fixed"][k] for s in steps) / len(steps) for k in range(len(grans))]

    def read(cost: list[float], acc: list[float]) -> float | None:
        value = symbols_at_accuracy(cost, acc, target)
        return value if math.isfinite(value) else None

    return {
        "fixed": read(grans, fixed),
        "forward": read(symbols, accuracy),
        "total_entropy": read([s + e / capacity for s, e in zip(symbols, entropy)], accuracy),
        "total_bitmap": read([s + b / capacity for s, b in zip(symbols, bitmap)], accuracy),
    }


def audit_comparison(runs: dict, reference: dict, target: float, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict | None:
    """The real-decoding saving of one receiver, with uncertainty.

    ``runs`` and ``reference`` map seed -> raw stream result (with the
    per-step audit ``live``) of the receiver and of the reference receiver
    (the frozen one) on the same streams. Every audited step was decoded by
    both, so steps are resampled in pairs, within each seed, ``n_boot`` times
    (the audited steps lie ``live_every`` steps apart, far beyond the
    correlation length of the stream). Readings are pooled over the seeds as
    in :func:`pooled_live`.

    Returns, for the receiver with refinement and signalling (entropy and
    bitmap) against the reference's fixed prefix, and against its own fixed
    prefix (what refinement adds): the point estimate, the percentile
    interval of the relative change, and the share of bootstrap draws in
    which the receiver's curve missed the target (an infinite cost: such
    draws stay in the sample, so a share above alpha/2 leaves the interval
    unbounded above, reported as ``None``). Also the per-seed readings, the input of
    a seed-level test. ``None`` if a seed or an audit is missing or the two
    receivers were not audited on the same steps."""
    seeds = sorted(runs)
    if not seeds or not set(seeds) <= set(reference):
        return None
    pairs = []
    for s in seeds:
        mine, theirs = runs[s].get("live"), reference[s].get("live")
        if not mine or not theirs or [r["step"] for r in mine] != [r["step"] for r in theirs]:
            return None
        pairs.append((mine, theirs))
    first = runs[seeds[0]]
    grans = [float(p["m"]) for p in first["live_fixed_curve"]]
    capacity = float(first["summary"]["feedback_capacity"])
    fallback = lambda group: [[p["feedback_entropy_bits"] for p in r["live_curve"]] for r in group]  # noqa: E731
    mine_fb, ref_fb = fallback([runs[s] for s in seeds]), fallback([reference[s] for s in seeds])

    def readings(index_sets: list) -> tuple[dict, dict]:
        mine, theirs, owners = [], [], []
        for k, ((a, b), idx) in enumerate(zip(pairs, index_sets)):
            mine += [a[i] for i in idx]
            theirs += [b[i] for i in idx]
            owners += [k] * len(idx)
        return _audit_reading(mine, grans, capacity, target, mine_fb, owners), _audit_reading(theirs, grans, capacity, target, ref_fb, owners)

    def contrasts(mine: dict, theirs: dict) -> dict:
        # a cost that never reaches the target is infinite; a reference that never does leaves the ratio undefined
        ratio = lambda a, b: None if b is None else (math.inf if a is None else a / b - 1)  # noqa: E731
        return {
            "vs_reference_entropy": ratio(mine["total_entropy"], theirs["fixed"]),
            "vs_reference_bitmap": ratio(mine["total_bitmap"], theirs["fixed"]),
            "vs_reference_fixed": ratio(mine["fixed"], theirs["fixed"]),
            "vs_own_fixed_entropy": ratio(mine["total_entropy"], mine["fixed"]),
            "vs_own_fixed_bitmap": ratio(mine["total_bitmap"], mine["fixed"]),
        }

    full = [list(range(len(a))) for a, _ in pairs]
    mine, theirs = readings(full)
    point = contrasts(mine, theirs)
    rng = np.random.default_rng(seed)
    draws = {key: [] for key in point}
    for _ in range(n_boot):
        values = contrasts(*readings([rng.integers(0, len(a), size=len(a)).tolist() for a, _ in pairs]))
        for key, value in values.items():
            draws[key].append(np.nan if value is None else value)
    out = {"seeds": seeds, "audited_steps": sum(len(a) for a, _ in pairs), "receiver": mine, "reference": theirs}
    for key, value in point.items():
        sample = np.asarray(draws[key], dtype=float)
        defined = sample[~np.isnan(sample)]
        low = float(np.quantile(defined, alpha / 2, method="lower")) if defined.size else None
        high = float(np.quantile(defined, 1 - alpha / 2, method="higher")) if defined.size else None
        out[key] = {
            "estimate": None if value is None or math.isinf(value) else value,  # None: not reached (or no reference)
            "ci_low": None if low is None or math.isinf(low) else low,
            "ci_high": None if high is None or math.isinf(high) else high,  # None: the interval is unbounded above
            "missed": float(np.isinf(sample).mean()),  # share of draws in which the receiver's curve missed the target
            "undefined": float(np.isnan(sample).mean()),  # share in which the reference's did
        }
    out["per_seed"] = {}
    for i, s in enumerate(seeds):
        alone = [list(range(len(a))) if k == i else [] for k, (a, _) in enumerate(pairs)]
        mine, theirs = readings(alone)
        out["per_seed"][s] = {"receiver": mine, "reference": theirs}
    return out
