"""Multi-timescale semantic-drift streams built from real corpora.

Three disjoint temporal bands, mirroring the fast/medium/slow structure of a
6G link:

* fast   - the channel: per-step SNR follows a clipped AR(1) process;
* medium - the topic of conversation: refreshed every ``tau_med`` steps;
* slow   - the knowledge domain (``hier``) or calendar time (``temporal``).

``hier``     Domains are visited in order for ``tau_slow`` steps each and the
             whole tour is repeated ``cycles`` times, so every domain is
             *revisited* - which is what makes retained knowledge measurable.
``temporal`` A dated corpus is replayed in chronological order. Nothing about
             the drift is designed by us: topic bursts and vocabulary change
             occur as they did in the world.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .corpora import Corpus


@dataclass
class StreamItem:
    t: int  # 1-indexed step
    texts: list[str]
    domain: str  # slow-band label (domain name or era index)
    topic: str  # medium-band label
    regime: int  # index of the slow regime (increments at every slow switch)
    snr_db: float
    regime_end: bool  # True on the last step of a slow regime


def snr_trace(n: int, cfg, rng: np.random.Generator) -> np.ndarray:
    """Clipped AR(1) SNR process (dB)."""
    mean, rho, sigma = float(cfg["mean"]), float(cfg["rho"]), float(cfg["sigma"])
    out = np.empty(n)
    value = mean
    for i in range(n):
        value = mean + rho * (value - mean) + sigma * np.sqrt(1.0 - rho**2) * rng.standard_normal()
        out[i] = float(np.clip(value, cfg["min"], cfg["max"]))
    return out


class HierStream:
    """Domain regimes (slow) containing topic regimes (medium)."""

    def __init__(self, corpora: dict[str, Corpus], scfg, seed: int):
        self.scfg = scfg
        self.rng = np.random.default_rng(seed)
        self.order = [d for _ in range(int(scfg["cycles"])) for d in scfg["domains"]]
        self.n_steps = len(self.order) * int(scfg["tau_slow"])
        self.pools: dict[str, dict[str, list[str]]] = {}
        for name in scfg["domains"]:
            by_topic: dict[str, list[str]] = {}
            for record in corpora[name].split("train"):
                by_topic.setdefault(record["topic"], []).append(record["text"])
            if not by_topic:
                raise RuntimeError(f"corpus '{name}' has no training sentences")
            self.pools[name] = by_topic
        self.snr = snr_trace(self.n_steps, scfg["snr"], self.rng)

    def __len__(self) -> int:
        return self.n_steps

    def __iter__(self) -> Iterator[StreamItem]:
        tau_slow, tau_med = int(self.scfg["tau_slow"]), int(self.scfg["tau_med"])
        kappa, batch = float(self.scfg["kappa"]), int(self.scfg["batch_size"])
        topic = ""
        for step in range(self.n_steps):
            regime = step // tau_slow
            domain = self.order[regime]
            pool = self.pools[domain]
            topics = sorted(pool)
            if (step % tau_slow) % tau_med == 0:  # medium-band refresh
                choices = [t for t in topics if t != topic] or topics
                topic = str(self.rng.choice(choices))
            texts = []
            for _ in range(batch):
                source = topic if self.rng.random() < kappa else str(self.rng.choice(topics))
                bucket = pool[source]
                texts.append(bucket[int(self.rng.integers(len(bucket)))])
            yield StreamItem(
                t=step + 1,
                texts=texts,
                domain=domain,
                topic=topic,
                regime=regime,
                snr_db=float(self.snr[step]),
                regime_end=(step + 1) % tau_slow == 0,
            )


class TemporalStream:
    """Chronological replay of a dated corpus, split into equal-size eras."""

    def __init__(self, corpus: Corpus, scfg, seed: int):
        self.scfg = scfg
        self.rng = np.random.default_rng(seed)
        records = [r for r in corpus.split("train") if r.get("time")]
        if not records:
            raise RuntimeError(f"corpus '{corpus.name}' has no dated training sentences")
        self.records = sorted(records, key=lambda r: r["time"])
        self.n_steps = int(scfg["temporal_steps"])
        self.n_eras = int(scfg["temporal_eras"])
        self.snr = snr_trace(self.n_steps, scfg["snr"], self.rng)

    def __len__(self) -> int:
        return self.n_steps

    def era_of_step(self, step: int) -> int:
        return min(step * self.n_eras // self.n_steps, self.n_eras - 1)

    def __iter__(self) -> Iterator[StreamItem]:
        batch = int(self.scfg["batch_size"])
        total = len(self.records)
        # a window that slides through time; sampling inside the window keeps
        # every batch contemporaneous while tolerating a short stream
        width = max(batch * 4, total // self.n_steps)
        for step in range(self.n_steps):
            centre = int((step + 0.5) * total / self.n_steps)
            lo = int(np.clip(centre - width // 2, 0, max(total - width, 0)))
            idx = self.rng.integers(lo, min(lo + width, total), size=batch)
            chosen = [self.records[int(i)] for i in idx]
            era = self.era_of_step(step)
            topics = [r["topic"] for r in chosen]
            yield StreamItem(
                t=step + 1,
                texts=[r["text"] for r in chosen],
                domain=f"era{era}",
                topic=max(set(topics), key=topics.count),
                regime=era,
                snr_db=float(self.snr[step]),
                regime_end=(step + 1 == self.n_steps) or self.era_of_step(step + 1) != era,
            )


def build_stream(corpora: dict[str, Corpus], scfg, seed: int):
    if scfg["kind"] == "hier":
        return HierStream(corpora, scfg, seed)
    if scfg["kind"] == "temporal":
        return TemporalStream(corpora[scfg["temporal_corpus"]], scfg, seed)
    raise ValueError(f"unknown stream kind '{scfg['kind']}'")


def build_probes(corpora: dict[str, Corpus], scfg, seed: int = 0) -> dict[str, list[str]]:
    """Held-out sentences per slow regime, used for the retention matrix."""
    rng = np.random.default_rng(10_000 + seed)
    size = int(scfg["probe_size"])
    probes: dict[str, list[str]] = {}
    if scfg["kind"] == "hier":
        for name in scfg["domains"]:
            texts = corpora[name].texts("test")
            pick = rng.permutation(len(texts))[:size]
            probes[name] = [texts[int(i)] for i in pick]
    else:
        corpus = corpora[scfg["temporal_corpus"]]
        dated = sorted((r for r in corpus.split("test") if r.get("time")), key=lambda r: r["time"])
        n_eras = int(scfg["temporal_eras"])
        for era in range(n_eras):
            chunk = dated[era * len(dated) // n_eras : (era + 1) * len(dated) // n_eras]
            pick = rng.permutation(len(chunk))[:size]
            probes[f"era{era}"] = [chunk[int(i)]["text"] for i in pick]
    return probes
