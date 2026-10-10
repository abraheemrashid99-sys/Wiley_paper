"""Multi-timescale semantic-drift streams built from real corpora.

A stream carries ``batch_size`` parallel *conversations*. At every step each
conversation sends the next sentence of its current document; when the
document ends it starts a new one. Consecutive messages of a conversation are
therefore related, exactly as in real traffic, and a receiver may use what it
has already decoded as context for what comes next.

Three disjoint temporal bands, mirroring the fast/medium/slow structure of a
6G link:

* fast   - the channel (per-step SNR follows a clipped AR(1) process) and the
           document a conversation is in;
* medium - the topic new documents are drawn from, refreshed every
           ``tau_med`` steps;
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
    texts: list[str]  # one message per conversation
    starts: list[bool]  # True where the message opens a new document (no context exists)
    domain: str  # slow-band label (domain name or era index)
    topic: str  # medium-band label
    regime: int  # index of the slow regime (increments at every slow switch)
    snr_db: float
    regime_end: bool  # True on the last step of a slow regime


@dataclass
class Probe:
    text: str
    history: list[str]  # the sentences of the same document that precede it (may be empty)


def snr_trace(n: int, cfg, rng: np.random.Generator) -> np.ndarray:
    """Clipped AR(1) SNR process (dB)."""
    mean, rho, sigma = float(cfg["mean"]), float(cfg["rho"]), float(cfg["sigma"])
    out = np.empty(n)
    value = mean
    for i in range(n):
        value = mean + rho * (value - mean) + sigma * np.sqrt(1.0 - rho**2) * rng.standard_normal()
        out[i] = float(np.clip(value, cfg["min"], cfg["max"]))
    return out


class _Conversations:
    """``n`` parallel conversations, each walking through one document."""

    def __init__(self, n: int):
        self.docs: list[list[str]] = [[] for _ in range(n)]
        self.pos = [0] * n

    def finished(self, slot: int) -> bool:
        return self.pos[slot] >= len(self.docs[slot])

    def open(self, slot: int, doc: list[str]) -> None:
        self.docs[slot], self.pos[slot] = doc, 0

    def next(self, slot: int) -> tuple[str, bool]:
        """The next sentence of the conversation and whether it opens a document."""
        position = self.pos[slot]
        self.pos[slot] += 1
        return self.docs[slot][position], position == 0


class HierStream:
    """Domain regimes (slow) containing topic regimes (medium)."""

    def __init__(self, corpora: dict[str, Corpus], scfg, seed: int):
        self.scfg = scfg
        self.rng = np.random.default_rng(seed)
        self.order = [d for _ in range(int(scfg["cycles"])) for d in scfg["domains"]]
        self.n_steps = len(self.order) * int(scfg["tau_slow"])
        self.pools: dict[str, dict[str, list[list[str]]]] = {}
        for name in scfg["domains"]:
            by_topic: dict[str, list[list[str]]] = {}
            for doc in corpora[name].documents("train"):
                by_topic.setdefault(doc[0]["topic"], []).append([r["text"] for r in doc])
            if not by_topic:
                raise RuntimeError(f"corpus '{name}' has no training sentences")
            self.pools[name] = by_topic
        self.snr = snr_trace(self.n_steps, scfg["snr"], self.rng)

    def __len__(self) -> int:
        return self.n_steps

    def __iter__(self) -> Iterator[StreamItem]:
        tau_slow, tau_med = int(self.scfg["tau_slow"]), int(self.scfg["tau_med"])
        kappa, batch = float(self.scfg["kappa"]), int(self.scfg["batch_size"])
        talks = _Conversations(batch)
        topic = ""
        for step in range(self.n_steps):
            regime = step // tau_slow
            domain = self.order[regime]
            pool = self.pools[domain]
            topics = sorted(pool)
            new_domain = step % tau_slow == 0
            if (step % tau_slow) % tau_med == 0:  # medium-band refresh
                choices = [t for t in topics if t != topic] or topics
                topic = str(self.rng.choice(choices))
            texts, starts = [], []
            for slot in range(batch):
                if new_domain or talks.finished(slot):  # a domain switch ends every conversation
                    source = topic if self.rng.random() < kappa else str(self.rng.choice(topics))
                    bucket = pool[source]
                    talks.open(slot, bucket[int(self.rng.integers(len(bucket)))])
                text, start = talks.next(slot)
                texts.append(text)
                starts.append(start)
            yield StreamItem(
                t=step + 1,
                texts=texts,
                starts=starts,
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
        docs = [doc for doc in corpus.documents("train") if doc[0].get("time")]
        if not docs:
            raise RuntimeError(f"corpus '{corpus.name}' has no dated training sentences")
        self.docs = sorted(docs, key=lambda doc: doc[0]["time"])
        self.n_steps = int(scfg["temporal_steps"])
        self.n_eras = int(scfg["temporal_eras"])
        self.snr = snr_trace(self.n_steps, scfg["snr"], self.rng)

    def __len__(self) -> int:
        return self.n_steps

    def era_of_step(self, step: int) -> int:
        return min(step * self.n_eras // self.n_steps, self.n_eras - 1)

    def __iter__(self) -> Iterator[StreamItem]:
        batch = int(self.scfg["batch_size"])
        total = len(self.docs)
        # a window that slides through time; new documents are drawn inside
        # it, which keeps every batch contemporaneous on a short stream
        width = max(batch * 4, total // self.n_steps)
        talks = _Conversations(batch)
        topic_of = [""] * batch
        for step in range(self.n_steps):
            centre = int((step + 0.5) * total / self.n_steps)
            lo = int(np.clip(centre - width // 2, 0, max(total - width, 0)))
            texts, starts = [], []
            for slot in range(batch):
                if talks.finished(slot):
                    doc = self.docs[int(self.rng.integers(lo, min(lo + width, total)))]
                    talks.open(slot, [r["text"] for r in doc])
                    topic_of[slot] = doc[0]["topic"]
                text, start = talks.next(slot)
                texts.append(text)
                starts.append(start)
            era = self.era_of_step(step)
            yield StreamItem(
                t=step + 1,
                texts=texts,
                starts=starts,
                domain=f"era{era}",
                topic=max(set(topic_of), key=topic_of.count),
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


def _probes_from(docs: list[list[dict]], size: int, rng: np.random.Generator) -> list[Probe]:
    """``size`` held-out sentences, each with the part of its document that
    precedes it, drawn from distinct positions of the given documents."""
    flat = [(doc, i) for doc in docs for i in range(len(doc))]
    picks = rng.permutation(len(flat))[:size]
    return [Probe(flat[int(j)][0][flat[int(j)][1]]["text"], [r["text"] for r in flat[int(j)][0][: flat[int(j)][1]]]) for j in picks]


def build_probes(corpora: dict[str, Corpus], scfg, seed: int = 0) -> dict[str, list[Probe]]:
    """Held-out sentences (with their document history) per slow regime, used
    for the retention matrix."""
    rng = np.random.default_rng(10_000 + seed)
    size = int(scfg["probe_size"])
    probes: dict[str, list[Probe]] = {}
    if scfg["kind"] == "hier":
        for name in scfg["domains"]:
            probes[name] = _probes_from(corpora[name].documents("test"), size, rng)
    else:
        corpus = corpora[scfg["temporal_corpus"]]
        dated = sorted((doc for doc in corpus.documents("test") if doc[0].get("time")), key=lambda doc: doc[0]["time"])
        n_eras = int(scfg["temporal_eras"])
        for era in range(n_eras):
            chunk = dated[era * len(dated) // n_eras : (era + 1) * len(dated) // n_eras]
            probes[f"era{era}"] = _probes_from(chunk, size, rng)
    return probes
