"""Fidelity and calibration metrics.

BLEU follows the convention of the text semantic-communication literature
(DeepSC): sentence-level BLEU-n with uniform weights over 1..n-grams and the
standard brevity penalty, averaged over sentences, without smoothing. Corpus
level sacreBLEU is reported alongside as the standardised reference.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import torch

_WORD = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def sentence_bleu(reference: str, hypothesis: str, n: int = 4) -> float:
    ref, hyp = words(reference), words(hypothesis)
    if not hyp or not ref:
        return 0.0
    log_sum = 0.0
    for k in range(1, n + 1):
        hyp_ngrams = _ngrams(hyp, k)
        total = sum(hyp_ngrams.values())
        if total == 0:
            return 0.0
        ref_ngrams = _ngrams(ref, k)
        overlap = sum(min(count, ref_ngrams[gram]) for gram, count in hyp_ngrams.items())
        if overlap == 0:
            return 0.0
        log_sum += math.log(overlap / total) / n
    brevity = 1.0 if len(hyp) >= len(ref) else math.exp(1.0 - len(ref) / len(hyp))
    return brevity * math.exp(log_sum)


def bleu_scores(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    out = {}
    for n in (1, 2, 3, 4):
        out[f"bleu{n}"] = float(np.mean([sentence_bleu(r, h, n) for r, h in zip(references, hypotheses)]))
    return out


def sacre_bleu(references: list[str], hypotheses: list[str]) -> float:
    import sacrebleu

    return float(sacrebleu.corpus_bleu(hypotheses, [references]).score)


def exact_match(references: list[str], hypotheses: list[str]) -> float:
    return float(np.mean([words(r) == words(h) for r, h in zip(references, hypotheses)]))


class SentenceSimilarity:
    """Cosine similarity of mean-pooled sentence embeddings.

    Uses a small sentence encoder (default all-MiniLM-L6-v2, 22 M parameters)
    through plain ``transformers``, which avoids an extra dependency.
    """

    def __init__(self, name: str, device: torch.device):
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(torch.float32).to(device).eval()
        self.device = device

    @torch.no_grad()
    def embed(self, texts: list[str], batch_size: int = 64) -> torch.Tensor:
        chunks = []
        for i in range(0, len(texts), batch_size):
            # an empty reconstruction must still yield an embedding
            piece = [t if t.strip() else "." for t in texts[i : i + batch_size]]
            enc = self.tok(piece, padding=True, truncation=True, max_length=64, return_tensors="pt").to(self.device)
            hidden = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)
            chunks.append(torch.nn.functional.normalize(pooled, dim=-1).cpu())
        return torch.cat(chunks)

    def __call__(self, references: list[str], hypotheses: list[str]) -> np.ndarray:
        return (self.embed(references) * self.embed(hypotheses)).sum(-1).numpy()


def load_similarity(name, device: torch.device):
    """Return a similarity model, or ``None`` when disabled or unavailable."""
    if not name:
        return None
    try:
        return SentenceSimilarity(name, device)
    except Exception as exc:  # offline machine: degrade loudly, not silently
        print(f"[metrics] sentence-similarity model '{name}' unavailable ({exc!r}); metric skipped")
        return None


def fidelity(references: list[str], hypotheses: list[str], sim=None) -> dict[str, float]:
    out = bleu_scores(references, hypotheses)
    out["sacrebleu"] = sacre_bleu(references, hypotheses)
    out["exact"] = exact_match(references, hypotheses)
    if sim is not None:
        out["sim"] = float(np.mean(sim(references, hypotheses)))
    return out


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = (confidence > lo) & (confidence <= hi) if lo > 0 else (confidence >= lo) & (confidence <= hi)
        if inside.any():
            ece += inside.mean() * abs(correct[inside].mean() - confidence[inside].mean())
    return float(ece)
