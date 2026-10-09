"""Multi-timescale (nested) optimisation with over-the-air synchronisation.

Following the continuum-memory rule of Nested Learning (Behrouz et al.,
NeurIPS 2025), every parameter *level* ``l`` has an update period ``C_l``:
gradients are accumulated at every step and applied once every ``C_l`` steps.

The twist that makes this a communication problem: a level that lives at the
*receiver* cannot be updated for free. The transmitter is the only node that
knows the source sentence, so it computes the update and must ship it. Each
receiver-side update is therefore quantised, its exact bit cost is recorded in
a ledger, and - crucially - the *quantised* update is applied at both ends, so
the two copies of the receiver parameters stay bit-identical at all times.
Between two updates of a level its parameters are constant, hence there is
never a stale-copy mismatch: staleness is expressed purely through ``C_l``.

Transmitter-side levels cost no air time and may run at any frequency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


@dataclass
class Level:
    name: str
    side: str  # "tx" (free to update) or "rx" (must be synchronised)
    params: list[tuple[str, torch.nn.Parameter]]
    period: int
    lr: float
    ema: bool = False

    @property
    def n_params(self) -> int:
        return sum(p.numel() for _, p in self.params)


def quantize_delta(delta: torch.Tensor, bits: int, topk: float = 1.0) -> tuple[torch.Tensor, int]:
    """Uniform symmetric quantisation of an update, with optional top-k
    sparsification. Returns ``(dequantised_delta, exact_cost_in_bits)``.

    Cost model: one 32-bit scale per tensor, ``bits`` per transmitted value
    and, when sparsified, ``ceil(log2 n)`` bits of position per kept value.
    An all-zero update costs a single flag bit.
    """
    flat = delta.detach().reshape(-1).to(torch.float32).cpu()
    n = flat.numel()
    if bits <= 0:  # idealised, unquantised reference (cost reported at 32 bit)
        return delta.detach().clone(), 32 * n
    scale = float(flat.abs().max())
    if scale == 0.0 or not math.isfinite(scale):
        return torch.zeros_like(delta), 1
    index_bits = 0
    if topk < 1.0:
        k = max(1, int(round(topk * n)))
        keep = torch.zeros(n, dtype=torch.bool)
        keep[flat.abs().topk(k).indices] = True
        flat = torch.where(keep, flat, torch.zeros_like(flat))
        index_bits = k * max(1, math.ceil(math.log2(n)))
        n_sent = k
    else:
        n_sent = n
    levels = 2 ** (bits - 1) - 1  # signed mid-tread quantiser
    if levels < 1:
        quant = torch.sign(flat) * scale  # 1-bit sign quantiser
    else:
        quant = torch.round(flat / scale * levels) / levels * scale
    cost = 32 + n_sent * bits + index_bits
    return quant.reshape(delta.shape).to(device=delta.device, dtype=delta.dtype), cost


@dataclass
class SyncLedger:
    """Exact account of every bit spent keeping the receiver aligned."""

    bits: dict[str, int] = field(default_factory=dict)
    events: dict[str, int] = field(default_factory=dict)
    trace: list[tuple[int, str, int]] = field(default_factory=list)  # (step, level, bits)

    def add(self, step: int, level: str, bits: int) -> None:
        self.bits[level] = self.bits.get(level, 0) + int(bits)
        self.events[level] = self.events.get(level, 0) + 1
        self.trace.append((int(step), level, int(bits)))

    @property
    def total_bits(self) -> int:
        return int(sum(self.bits.values()))

    def cumulative(self, n_steps: int) -> list[int]:
        out, running, i = [], 0, 0
        trace = sorted(self.trace)
        for step in range(1, n_steps + 1):
            while i < len(trace) and trace[i][0] <= step:
                running += trace[i][2]
                i += 1
            out.append(running)
        return out

    def summary(self) -> dict:
        return {"total_bits": self.total_bits, "bits": dict(self.bits), "events": dict(self.events)}


class NestedOptimizer:
    """One Adam optimiser per level, chunked updates, EMA anchor, sync ledger."""

    def __init__(self, levels: list[Level], sync_bits: int = 8, sync_topk: float = 1.0, ema_decay: float = 0.99, ema_beta: float = 0.9, grad_clip: float = 1.0):
        self.levels = [lv for lv in levels if lv.params]
        self.sync_bits, self.sync_topk = int(sync_bits), float(sync_topk)
        self.ema_decay, self.ema_beta = float(ema_decay), float(ema_beta)
        self.grad_clip = float(grad_clip)
        self.ledger = SyncLedger()
        self._opt, self._acc, self._count, self._ema = {}, {}, {}, {}
        for lv in self.levels:
            tensors = [p for _, p in lv.params]
            self._opt[lv.name] = torch.optim.Adam(tensors, lr=lv.lr)
            self._acc[lv.name] = [torch.zeros_like(p) for p in tensors]
            self._count[lv.name] = 0
            if lv.ema:
                self._ema[lv.name] = [p.detach().clone() for p in tensors]

    def trainable(self) -> list[torch.nn.Parameter]:
        return [p for lv in self.levels for _, p in lv.params]

    def expected_bits_per_update(self, level: Level) -> int:
        """Cost of one dense synchronisation of ``level`` (0 for tx levels)."""
        if level.side != "rx":
            return 0
        total = 0
        for _, p in level.params:
            n = p.numel()
            if self.sync_bits <= 0:
                total += 32 * n
            elif self.sync_topk < 1.0:
                k = max(1, int(round(self.sync_topk * n)))
                total += 32 + k * (self.sync_bits + max(1, math.ceil(math.log2(n))))
            else:
                total += 32 + n * self.sync_bits
        return total

    @torch.no_grad()
    def step(self, t: int) -> None:
        """Call once per stream step, after ``loss.backward()`` (t is 1-indexed)."""
        for lv in self.levels:
            tensors = [p for _, p in lv.params]
            for acc, p in zip(self._acc[lv.name], tensors):
                if p.grad is not None:
                    acc.add_(p.grad)
            self._count[lv.name] += 1
            if t % lv.period != 0:
                continue
            count = self._count[lv.name]
            for acc, p in zip(self._acc[lv.name], tensors):
                p.grad = acc / count
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(tensors, self.grad_clip)
            before = [p.detach().clone() for p in tensors] if lv.side == "rx" else None
            self._opt[lv.name].step()
            if lv.ema:  # continuum-memory anchor on the slow level
                for ema, p in zip(self._ema[lv.name], tensors):
                    ema.mul_(self.ema_decay).add_(p, alpha=1.0 - self.ema_decay)
                    p.mul_(self.ema_beta).add_(ema, alpha=1.0 - self.ema_beta)
            if before is not None:  # ship the quantised update; apply the same at both ends
                bits = 0
                for old, p in zip(before, tensors):
                    delta_q, cost = quantize_delta(p - old, self.sync_bits, self.sync_topk)
                    p.copy_(old + delta_q)
                    bits += cost
                self.ledger.add(t, lv.name, bits)
            for acc in self._acc[lv.name]:
                acc.zero_()
            self._count[lv.name] = 0
        for lv in self.levels:
            for _, p in lv.params:
                p.grad = None
