"""Sentence extraction and cleaning shared by every corpus loader."""

from __future__ import annotations

import html
import re

_WS = re.compile(r"\s+")
_SPACE_PUNCT = re.compile(r" ([.,;:!?])(?= |$)")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
_WIRE_TAG = re.compile(r"\((?:Reuters|AP|AFP|SPACE\.com|washingtonpost\.com|CNN|Forbes\.com|USATODAY\.com)\)", re.I)
_WIRE_LEAD = re.compile(r"^(?:[A-Z][A-Za-z.,' ]{0,40}\s)?(?:\(?(?:Reuters|AP|AFP)\)?)\s+-\s+")
_BRACKET_TAG = re.compile(r"\s*\[[^\]]{0,40}\]\s*$")


def clean(text: str) -> str:
    """Normalise whitespace and undo the escaping artefacts of news dumps."""
    text = html.unescape(text.replace("#39;", "'").replace("#36;", "$"))
    text = text.replace("\\", " ")
    text = _WIRE_TAG.sub(" ", text)
    text = _WIRE_LEAD.sub("", text.strip())
    text = _SPACE_PUNCT.sub(r"\1", _WS.sub(" ", text))  # "Pipeline ." left behind by a removed tag
    return text.strip()


def strip_bracket_tag(text: str) -> str:
    """Drop a trailing ``[3GPP Release 17]``-style source tag."""
    return _BRACKET_TAG.sub("", text).strip()


def split_sentences(text: str) -> list[str]:
    text = clean(text)
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def n_words(text: str) -> int:
    return len(text.split())


def keep_sentence(text: str, min_words: int, max_words: int) -> bool:
    """The DeepSC length protocol (4-30 words) plus basic hygiene filters."""
    words = n_words(text)
    if words < min_words or words > max_words:
        return False
    if "http" in text or "<" in text or ">" in text or "&lt" in text:
        return False
    letters = sum(ch.isalpha() for ch in text)
    return letters >= 0.6 * len(text.replace(" ", ""))
