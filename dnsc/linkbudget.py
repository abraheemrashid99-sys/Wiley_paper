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
    (entropy and bitmap), each ``None`` if the pooled curve never reaches the
    target, and the number of audited steps."""
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
    return {
        "fixed": read([float(p["m"]) for p in fixed[0]], mean_curve(fixed, "tok_acc")),
        "forward": read(mean_curve(curves, "symbols"), accuracy),
        "total_entropy": read(mean_curve(curves, "symbols_total_entropy"), accuracy),
        "total_bitmap": read(mean_curve(curves, "symbols_total"), accuracy),
        "audited_steps": sum(int(r["summary"].get("live_steps", 0)) for r in runs),
    }
