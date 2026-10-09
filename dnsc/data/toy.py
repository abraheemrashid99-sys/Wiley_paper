"""Synthetic toy corpora.

These exist ONLY so that the unit tests and the offline smoke run can exercise
the full pipeline without any download. They are template-generated, carry no
scientific meaning, and must never be used for a reported result.
"""

from __future__ import annotations

import random

_DOMAINS = {
    "toy_a": {  # parliamentary register
        "debate": (
            ["the council", "the commission", "this parliament", "the committee", "our group"],
            ["supports", "rejects", "debates", "amends", "adopts", "welcomes"],
            ["the proposal", "the directive", "the budget", "the resolution", "the report"],
            ["on agriculture", "on fisheries", "on trade", "on human rights", "on energy policy"],
        ),
        "procedure": (
            ["the president", "the rapporteur", "the member", "the chairman"],
            ["opens", "closes", "postpones", "announces", "interrupts"],
            ["the sitting", "the vote", "the session", "the agenda", "the minutes"],
            ["until tomorrow", "this afternoon", "after the break", "at noon"],
        ),
    },
    "toy_b": {  # news register, two topics
        "sports": (
            ["the striker", "the captain", "the home team", "the champion", "the coach"],
            ["scores", "wins", "loses", "celebrates", "misses", "defends"],
            ["the final", "the match", "the title", "a late goal", "the penalty"],
            ["in extra time", "before a full stadium", "against the rivals", "this season"],
        ),
        "business": (
            ["the bank", "the investor", "the company", "the market", "the regulator"],
            ["raises", "cuts", "reports", "forecasts", "approves", "delays"],
            ["interest rates", "quarterly profits", "the merger", "share prices", "the dividend"],
            ["amid inflation fears", "after strong earnings", "for the third quarter", "next year"],
        ),
    },
    "toy_c": {  # telecom register, two topics
        "radio": (
            ["the base station", "the scheduler", "the user equipment", "the antenna array"],
            ["allocates", "measures", "reports", "estimates", "retransmits", "beamforms"],
            ["resource blocks", "the channel state", "the pilot signal", "the uplink grant"],
            ["in every slot", "under fading", "at the cell edge", "per subframe"],
        ),
        "core": (
            ["the network function", "the session manager", "the slice controller", "the gateway"],
            ["registers", "authenticates", "releases", "forwards", "establishes", "monitors"],
            ["the data session", "the subscriber profile", "the bearer", "the policy rule"],
            ["over the service interface", "during handover", "for each slice", "on request"],
        ),
    },
}

_FILLERS = ["", "", "today", "again", "as expected", "without delay", "once more"]


def toy_names() -> list[str]:
    return sorted(_DOMAINS)


def make_toy(name: str, n_docs: int = 900, seed: int = 0) -> list[dict]:
    """Return raw records ``{text, topic, time, doc}`` for a toy domain."""
    rng = random.Random(f"{name}:{seed}")
    topics = _DOMAINS[name]
    records = []
    for doc in range(n_docs):
        topic = rng.choice(sorted(topics))
        subj, verb, obj, tail = topics[topic]
        parts = [rng.choice(subj), rng.choice(verb), rng.choice(obj), rng.choice(tail), rng.choice(_FILLERS)]
        text = " ".join(p for p in parts if p)
        records.append({"text": text[0].upper() + text[1:] + ".", "topic": topic, "time": f"{2000 + doc * 10 // n_docs}", "doc": doc})
    return records
