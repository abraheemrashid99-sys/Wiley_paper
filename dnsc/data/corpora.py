"""Corpus loaders.

Every loader returns sentence-level records ``{text, topic, time, doc}``.
Records are filtered with the DeepSC length protocol, de-duplicated, capped,
split by *document* (so two sentences of one article never straddle a split)
and cached as JSON lines under ``<cache_dir>/processed``.

Real corpora
------------
europarl   Europarl v7, English side (Koehn, 2005) - the reference corpus of
           text semantic communication since DeepSC. Hugging Face, streamed.
ag_news    AG News (Zhang et al., 2015) - four topic classes. GitHub CSV,
           Hugging Face fallback.
huffpost   News Category Dataset (Misra, 2022) - 2012-2022 headlines with
           dates; the natural temporal-shift corpus of Wild-Time. Hugging Face.
teleqna    TeleQnA (Maatouk et al., 2023) - telecom/3GPP question bank.
           Password-protected archive from the authors' GitHub repository.

The same two corpora from fixed GitHub sources, for machines that cannot reach
Hugging Face. They are separate names because they are different subsets:

europarl_nltk  the Europarl sample distributed with NLTK: ten sittings of
               January-February 2000, about 20 000 sentences.
huffpost_gh    a GitHub mirror of the News Category Dataset v3 that holds
               186 041 of the 209 527 records (5 Oct 2012 - 23 Sep 2022).

Both downloads are checked against a SHA-256 digest.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
import urllib.request
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import toy
from .text import clean, keep_sentence, split_sentences, strip_bracket_tag

CACHE_VERSION = "v3"  # bump when the cleaning rules change, so stale caches are rebuilt
AG_NEWS_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/{split}.csv"
AG_NEWS_TOPICS = {1: "world", 2: "sports", 3: "business", 4: "scitech"}
TELEQNA_URL = "https://raw.githubusercontent.com/netop-team/TeleQnA/main/TeleQnA.zip"
TELEQNA_PASSWORD = b"teleqnadataset"  # published by the authors in their README
EUROPARL_NLTK_URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/europarl_raw.zip"
EUROPARL_NLTK_SHA256 = "ad553e177baac263840c10980e6f3e76d5d15f7f7a078bd98520b36edb69b27c"
HUFFPOST_GH_URL = "https://raw.githubusercontent.com/roahn-w/NLP-News-Category-Classifier/main/News_Category_Dataset_v3%20copy.json.zip"
HUFFPOST_GH_SHA256 = "7f9adc50a461dadbacf2e076a1e87822d33fd9fdffd15af028c28f81c3505270"


@dataclass
class Corpus:
    name: str
    records: list[dict] = field(default_factory=list)

    def split(self, which: str) -> list[dict]:
        return [r for r in self.records if r["split"] == which]

    def texts(self, which: str) -> list[str]:
        return [r["text"] for r in self.split(which)]

    def documents(self, which: str) -> list[list[dict]]:
        """The sentences of a split grouped into their documents, in order."""
        docs: list[list[dict]] = []
        last = None
        for record in self.split(which):
            if record["doc"] != last:
                docs.append([])
                last = record["doc"]
            docs[-1].append(record)
        return docs

    @property
    def topics(self) -> list[str]:
        return sorted({r["topic"] for r in self.records})

    def summary(self) -> dict:
        words = [len(r["text"].split()) for r in self.records]
        times = sorted(r["time"] for r in self.records if r.get("time"))
        return {
            "corpus": self.name,
            "sentences": len(self.records),
            "train": len(self.split("train")),
            "val": len(self.split("val")),
            "test": len(self.split("test")),
            "topics": len(self.topics),
            "mean_words": round(sum(words) / max(len(words), 1), 2),
            "time_span": f"{times[0]}..{times[-1]}" if times else "-",
        }


# --------------------------------------------------------------------------
# raw loaders: each returns a list of {text (raw, may be multi-sentence), topic, time, doc}
# --------------------------------------------------------------------------


def _download(url: str, timeout: int = 120) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "dnsc/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _raw_ag_news(max_docs: int) -> list[dict]:
    try:
        payload = _download(AG_NEWS_URL.format(split="train")).decode("utf-8")
        rows = csv.reader(io.StringIO(payload))
        out = []
        for doc, row in enumerate(rows):
            if len(row) != 3:
                continue
            out.append({"text": f"{row[1]}. {row[2]}", "topic": AG_NEWS_TOPICS[int(row[0])], "time": "", "doc": doc})
        if out:
            return out
    except Exception as exc:  # fall through to Hugging Face
        print(f"[data] ag_news: GitHub CSV unavailable ({exc!r}); trying Hugging Face")
    from datasets import load_dataset

    ds = load_dataset("fancyzhx/ag_news", split="train")
    names = ["world", "sports", "business", "scitech"]
    return [{"text": r["text"], "topic": names[int(r["label"])], "time": "", "doc": i} for i, r in enumerate(ds)]


def _raw_teleqna(max_docs: int) -> list[dict]:
    import pyzipper

    blob = _download(TELEQNA_URL)
    with pyzipper.AESZipFile(io.BytesIO(blob)) as archive:
        member = next(n for n in archive.namelist() if n.lower().endswith((".txt", ".json")))
        data = json.loads(archive.read(member, pwd=TELEQNA_PASSWORD).decode("utf-8"))
    out = []
    for doc, (_, item) in enumerate(sorted(data.items(), key=lambda kv: int(kv[0].split()[-1]))):
        topic = str(item.get("category", "all")).strip().lower().replace(" ", "_")
        pieces = [strip_bracket_tag(str(item.get("question", "")))]
        if item.get("explanation"):
            pieces.append(str(item["explanation"]))
        answer = str(item.get("answer", ""))
        if ":" in answer:
            pieces.append(answer.split(":", 1)[1].strip())
        for piece in pieces:
            out.append({"text": piece, "topic": topic, "time": "", "doc": doc})
    return out


def _raw_huffpost(max_docs: int) -> list[dict]:
    from datasets import load_dataset

    return _huffpost_rows(load_dataset("heegyu/news-category-dataset", split="train"))


def _raw_europarl(max_docs: int) -> list[dict]:
    from datasets import load_dataset

    # Smallest English-containing pair; streamed, so only the rows that are
    # actually consumed are downloaded.
    ds = load_dataset("Helsinki-NLP/europarl", "en-ro", split="train", streaming=True)
    want = max_docs * 3  # head-room for the length filter
    out = []
    for i, row in enumerate(ds):
        if i >= want:
            break
        text = row["translation"]["en"]
        # consecutive lines belong to one speech: group 50 lines per "document"
        out.append({"text": text, "topic": "all", "time": "", "doc": i // 50})
    return out


def _download_checked(url: str, digest: str) -> bytes:
    blob = _download(url, timeout=300)
    found = hashlib.sha256(blob).hexdigest()
    if found != digest:
        raise RuntimeError(f"unexpected content at {url} (sha256 {found})")
    return blob


def _huffpost_rows(rows) -> list[dict]:
    out = []
    for doc, row in enumerate(rows):
        day = str(row.get("date", ""))[:10]
        topic = str(row.get("category", "all")).strip().lower().replace(" ", "_")
        for piece in (row.get("headline") or "", row.get("short_description") or ""):
            if piece:
                out.append({"text": piece, "topic": topic, "time": day, "doc": doc})
    return out


def _raw_huffpost_gh(max_docs: int) -> list[dict]:
    import zipfile

    with zipfile.ZipFile(io.BytesIO(_download_checked(HUFFPOST_GH_URL, HUFFPOST_GH_SHA256))) as archive:
        member = next(n for n in archive.namelist() if n.endswith(".json") and not n.startswith("__MACOSX"))
        lines = archive.read(member).decode("utf-8").splitlines()
    return _huffpost_rows(json.loads(line) for line in lines if line.strip())


_DETOK = [
    (re.compile(r"\s+([,.;:!?%)\]])"), r"\1"),  # no space before closing punctuation
    (re.compile(r"([(\[])\s+"), r"\1"),  # none after an opening bracket
    (re.compile(r"(\w)\s+'\s+s\b"), r"\1's"),  # "Mr Jonckheer ' s" -> "Mr Jonckheer's"
    (re.compile(r"(\w)\s+'s\b"), r"\1's"),
    (re.compile(r"(\w)\s+n't\b"), r"\1n't"),
]


def detokenize(text: str) -> str:
    """Undo the space-separated punctuation of pre-tokenised corpora."""
    for pattern, replacement in _DETOK:
        text = pattern.sub(replacement, text)
    return text


def _raw_europarl_nltk(max_docs: int) -> list[dict]:
    import zipfile

    out = []
    with zipfile.ZipFile(io.BytesIO(_download_checked(EUROPARL_NLTK_URL, EUROPARL_NLTK_SHA256))) as archive:
        members = sorted(n for n in archive.namelist() if n.startswith("europarl_raw/english/") and n.endswith(".en"))
        line_no = 0
        for member in members:  # one file per sitting, one sentence per line
            for line in archive.read(member).decode("utf-8", errors="replace").splitlines():
                if line.strip():
                    # consecutive lines belong to one speech: group 50 lines per "document"
                    out.append({"text": detokenize(line.strip()), "topic": "all", "time": "", "doc": line_no // 50})
                    line_no += 1
    return out


def _raw_toy(name: str) -> Callable[[int], list[dict]]:
    return lambda max_docs: toy.make_toy(name)


_RAW: dict[str, Callable[[int], list[dict]]] = {
    "ag_news": _raw_ag_news,
    "teleqna": _raw_teleqna,
    "huffpost": _raw_huffpost,
    "europarl": _raw_europarl,
    "europarl_nltk": _raw_europarl_nltk,
    "huffpost_gh": _raw_huffpost_gh,
    **{name: _raw_toy(name) for name in toy.toy_names()},
}


def available_corpora() -> list[str]:
    return sorted(_RAW)


# --------------------------------------------------------------------------
# processing
# --------------------------------------------------------------------------


def assign_split(name: str, doc: int) -> str:
    """Deterministic 80/10/10 split keyed on the document id."""
    bucket = zlib.crc32(f"{name}:{doc}".encode()) % 10
    return "train" if bucket < 8 else ("val" if bucket == 8 else "test")


def process(name: str, raw: list[dict], min_words: int, max_words: int, max_sentences: int) -> list[dict]:
    seen: set[str] = set()
    records = []
    for item in raw:
        for sentence in split_sentences(item["text"]):
            sentence = clean(sentence)
            key = sentence.lower()
            if key in seen or not keep_sentence(sentence, min_words, max_words):
                continue
            seen.add(key)
            records.append(
                {
                    "text": sentence,
                    "topic": item["topic"],
                    "time": item.get("time", ""),
                    "doc": int(item["doc"]),
                    "split": assign_split(name, int(item["doc"])),
                }
            )
    if len(records) > max_sentences:  # keep whole documents, so that consecutive sentences stay together
        doc_ids = sorted({r["doc"] for r in records})
        random.Random(0).shuffle(doc_ids)
        size: dict[int, int] = {}
        for r in records:
            size[r["doc"]] = size.get(r["doc"], 0) + 1
        keep, total = set(), 0
        for doc in doc_ids:
            if total >= max_sentences:
                break
            keep.add(doc)
            total += size[doc]
        records = [r for r in records if r["doc"] in keep]
    return records


def get_corpus(name: str, data_cfg, refresh: bool = False) -> Corpus:
    """Load a corpus from the local cache, building the cache on first use."""
    if name not in _RAW:
        raise KeyError(f"unknown corpus '{name}'; available: {available_corpora()}")
    cache = Path(data_cfg["cache_dir"]) / "processed"
    tag = f"{name}_w{data_cfg['min_words']}-{data_cfg['max_words']}_n{data_cfg['max_sentences']}_{CACHE_VERSION}"
    path = cache / f"{tag}.jsonl"
    if path.exists() and not refresh:
        with open(path, "r", encoding="utf-8") as handle:
            return Corpus(name, [json.loads(line) for line in handle])
    print(f"[data] building corpus '{name}' (first use; cached afterwards)")
    raw = _RAW[name](int(data_cfg["max_sentences"]))
    records = process(name, raw, data_cfg["min_words"], data_cfg["max_words"], data_cfg["max_sentences"])
    if not records:
        raise RuntimeError(f"corpus '{name}' produced no sentences after filtering")
    cache.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return Corpus(name, records)
