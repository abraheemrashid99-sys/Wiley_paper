"""Physical channel models and the shell-wise power constraint.

Symbols are real-valued; ``m`` real symbols correspond to ``m/2`` complex
channel uses. SNR is the per-symbol ratio of signal power (normalised to one)
to noise variance.

Shell-wise normalisation
------------------------
With granularities ``m_1 < m_2 < ... < m_K`` the payload is cut into shells
``(m_{k-1}, m_k]`` and every shell is scaled to unit average power on its own.
Any prefix is then a union of unit-power shells, so (i) every operating point
meets the same average-power constraint exactly and (ii) sending shell ``k+1``
after shells ``1..k`` is *identical* to having sent the prefix ``m_{k+1}`` in
one go. Property (ii) is what makes incremental refinement exact.
"""

from __future__ import annotations

import math

import torch


def shell_slices(granularities: list[int]) -> list[slice]:
    edges = [0] + list(granularities)
    return [slice(edges[i], edges[i + 1]) for i in range(len(granularities))]


def shell_normalize(z: torch.Tensor, granularities: list[int]) -> torch.Tensor:
    """Scale every shell of ``z`` (B, L) to unit average power per symbol."""
    parts = []
    for sl in shell_slices(granularities):
        shell = z[:, sl]
        width = sl.stop - sl.start
        norm = shell.norm(dim=1, keepdim=True).clamp_min(1e-8)
        parts.append(shell * (math.sqrt(width) / norm))
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

    def __call__(self, x: torch.Tensor, snr_db) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(y, effective_snr_db)`` for unit-power symbols ``x`` (B, m)."""
        eff = self.effective_snr_db(x.shape[0], snr_db)
        sigma = torch.pow(10.0, -eff / 20.0).unsqueeze(1)
        noise = torch.randn(x.shape, generator=self.generator) * sigma
        return x + noise.to(device=x.device, dtype=x.dtype), eff


def capacity_real(snr_db) -> float | torch.Tensor:
    """Shannon capacity in bits per *real* channel use, 0.5 log2(1 + SNR)."""
    if isinstance(snr_db, torch.Tensor):
        return 0.5 * torch.log2(1.0 + torch.pow(10.0, snr_db / 10.0))
    return 0.5 * math.log2(1.0 + 10.0 ** (float(snr_db) / 10.0))
