"""Separation-based reference systems (derived, not simulated bit-by-bit).

A fair comparison for a semantic codec is not a weak source coder: it is the
best that Shannon's separation theorem allows. We therefore compute, for each
sentence, the number of source bits under two lossless coders -

* ``deflate``  raw DEFLATE of the UTF-8 bytes (a practical, universal coder);
* ``lm``       the ideal arithmetic-coding length under the *same* frozen LLM
               that the semantic codec uses, ``sum -log2 p(token | prefix)``.
               This is the strongest classical opponent: it enjoys exactly the
               same language prior as the semantic system.

- and an ideal channel code, in two versions. With a budget of ``m`` real
symbols at post-equalisation SNR ``g`` the message is delivered iff its bits
fit, and is lost otherwise (outage):

* asymptotic          ``bits <= m * C``, with ``C = 0.5 log2(1 + g)``. Capacity
                      is only approached at infinite block length, so at the
                      16-256 symbols used here this is generous to separation.
* finite block length ``bits <= m*C - sqrt(m*V) * Qinv(eps) + 0.5*log2(m)``,
                      the normal approximation of Polyanskiy, Poor and Verdu
                      (2010) with dispersion ``V`` and block error rate
                      ``eps``. This is the fair reference at short blocks.

Both are upper bounds for a real separate source-channel design at that
budget and are labelled "derived" wherever they are reported.
"""

from __future__ import annotations

import zlib

import numpy as np
import torch
import torch.nn.functional as F

from .channel import capacity_real


def deflate_bits(text: str) -> int:
    compressor = zlib.compressobj(9, zlib.DEFLATED, -15)  # raw stream: no header/checksum
    return 8 * len(compressor.compress(text.encode("utf-8")) + compressor.flush())


@torch.no_grad()
def lm_bits(codec, texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Ideal code length (bits) of each sentence, EOS included, under the
    frozen backbone with no adapter active."""
    codec._side(None)
    start = codec.tok.bos_token_id if codec.tok.bos_token_id is not None else codec.tok.eos_token_id
    out = []
    for i in range(0, len(texts), batch_size):
        batch = codec.tokenize(texts[i : i + batch_size])
        ids, lens = batch.ids, batch.lens
        bsz, width = ids.shape
        context = torch.cat([torch.full((bsz, 1), start, dtype=torch.long, device=ids.device), ids[:, :-1]], dim=1)
        mask = (torch.arange(width, device=ids.device).unsqueeze(0) < lens.unsqueeze(1)).long()
        hidden = codec.lm.base_model(input_ids=context, attention_mask=mask).last_hidden_state
        logp = F.log_softmax(codec._head(hidden).float(), dim=-1)
        token_lp = logp.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
        nats = -(token_lp * mask.to(token_lp.dtype)).sum(1)
        out.append((nats / np.log(2.0)).cpu().numpy())
    return np.concatenate(out)


def max_bits(m: int, eff_snr_db, eps: float | None = None) -> np.ndarray:
    """Largest payload (bits) an ideal code carries in ``m`` real symbols.

    ``eps=None`` gives the asymptotic limit ``m*C``; otherwise the normal
    approximation at block error rate ``eps``, clipped to ``[0, m*C]``."""
    snr_db = np.asarray(eff_snr_db, dtype=float)
    capacity = capacity_real(torch.as_tensor(snr_db, dtype=torch.float64)).numpy()
    if eps is None:
        return m * capacity
    from scipy.stats import norm

    p = 10.0 ** (snr_db / 10.0)
    dispersion = p * (p + 2.0) / (2.0 * (p + 1.0) ** 2) * np.log2(np.e) ** 2  # bits^2 per real use
    approx = m * capacity - np.sqrt(m * dispersion) * norm.isf(eps) + 0.5 * np.log2(m)
    # the approximation is loose for very short blocks: keep it inside [0, m*C]
    return np.clip(approx, 0.0, m * capacity)


def delivered(bits: np.ndarray, m: int, eff_snr_db, eps: float | None = None) -> np.ndarray:
    """Boolean mask: which messages fit in ``m`` symbols at the given SNRs."""
    return np.asarray(bits, dtype=float) <= max_bits(m, eff_snr_db, eps)
