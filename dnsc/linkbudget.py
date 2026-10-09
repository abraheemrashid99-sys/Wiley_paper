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
