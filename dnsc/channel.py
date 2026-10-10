"""Physical channel models and the nested power constraint.

Symbols are real-valued; ``m`` real symbols correspond to ``m/2`` complex
channel uses. SNR is the ratio of the average signal power per symbol
(normalised to one) to the noise variance.

Nested average-power constraint
-------------------------------
With granularities ``m_1 < m_2 < ... < m_K`` every codeword is cut into shells
``(m_{k-1}, m_k]``. Each shell is centred and scaled so that its *average*
power per symbol, taken over the source distribution of the codewords, is at
most one. Consequences:

(i)  every prefix meets the same average-power constraint, so all operating
     points are compared at equal transmit power;
(ii) sending shell ``k+1`` after shells ``1..k`` is *identical* to having sent
     the prefix ``m_{k+1}`` in one go, which makes incremental refinement
     exact.

The codebook constraint holds on average over the source distribution it was
computed on. To make the transmit power *exact* for every message whatever
its content, each transmitted block (one shell of one message) is additionally
scaled to unit average power per symbol. The scale is applied to the whole
block including its pilots, so it is absorbed in the channel estimate and the
receiver, which is assumed to have perfect channel state information, sees it
only as a change of effective SNR for that block.
"""

from __future__ import annotations

import math

import torch


def shell_slices(granularities: list[int]) -> list[slice]:
    edges = [0] + list(granularities)
    return [slice(edges[i], edges[i + 1]) for i in range(len(granularities))]


def nested_power_normalize(raw: torch.Tensor, weights: torch.Tensor, granularities: list[int], peak: float = 4.0) -> torch.Tensor:
    """Normalise a codebook ``raw`` (V, M) shell by shell.

    ``weights`` (V,) is the probability of each codeword. After the call every
    shell has zero weighted mean, weighted average power per symbol <= 1 and
    per-codeword power per symbol <= ``peak``."""
    weights = (weights / weights.sum()).to(raw.dtype).unsqueeze(1)
    parts = []
    for sl in shell_slices(granularities):
        shell = raw[:, sl]
        width = sl.stop - sl.start
        shell = shell - (weights * shell).sum(0, keepdim=True)
        power = (weights * shell.pow(2)).sum().clamp_min(1e-12) / width
        shell = shell / power.sqrt()
        norm = shell.norm(dim=1, keepdim=True).clamp_min(1e-12)
        limit = math.sqrt(peak * width)
        parts.append(shell * torch.clamp(limit / norm, max=1.0))
    return torch.cat(parts, dim=1)


class Channel:
    """AWGN or Rayleigh block fading with perfect CSI at the receiver.

    Noise is drawn on the CPU from a private generator and then moved to the
    signal's device, so a given seed yields the same realisation on CPU, CUDA
    and Apple MPS.
    """

    MIN_GAIN = 0.05  # floor on |h| under zero-forcing equalisation (deep-fade clip)

    def __init__(self, kind: str = "awgn", seed: int = 0):
        if kind not in ("awgn", "rayleigh"):
            raise ValueError(f"unknown channel '{kind}'")
        self.kind = kind
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def reseed(self, seed: int) -> None:
        self.generator.manual_seed(seed)

    def effective_snr_db(self, batch: int, snr_db) -> torch.Tensor:
        """Per-message post-equalisation SNR in dB (CPU tensor of shape (B,))."""
        snr = torch.as_tensor(snr_db, dtype=torch.float32).reshape(-1).cpu()
        if snr.numel() == 1:
            snr = snr.expand(batch).clone()
        if self.kind == "rayleigh":
            # |h|^2 ~ Exp(1) for h ~ CN(0, 1); one draw per message (block fading)
            gain2 = torch.empty(batch).exponential_(1.0, generator=self.generator)
            gain2 = gain2.clamp_min(self.MIN_GAIN**2)
            snr = snr + 10.0 * torch.log10(gain2)
        return snr

    def noise(self, shape, std: torch.Tensor) -> torch.Tensor:
        """Gaussian noise of ``shape`` with standard deviation ``std``
        (broadcastable CPU tensor), drawn from the private generator."""
        return torch.randn(tuple(shape), generator=self.generator) * std

    def __call__(self, x: torch.Tensor, snr_db) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(y, effective_snr_db)`` for unit-power symbols ``x`` (B, m)."""
        eff = self.effective_snr_db(x.shape[0], snr_db)
        noise = self.noise(x.shape, torch.pow(10.0, -eff / 20.0).unsqueeze(1))
        return x + noise.to(device=x.device, dtype=x.dtype), eff


def capacity_real(snr_db) -> float | torch.Tensor:
    """Shannon capacity in bits per *real* channel use, 0.5 log2(1 + SNR)."""
    if isinstance(snr_db, torch.Tensor):
        return 0.5 * torch.log2(1.0 + torch.pow(10.0, snr_db / 10.0))
    return 0.5 * math.log2(1.0 + 10.0 ** (float(snr_db) / 10.0))
