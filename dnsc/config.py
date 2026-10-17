"""Configuration loading.

A config is a plain nested dict loaded from YAML, wrapped in :class:`Cfg` for
attribute access. Every script accepts ``--config path.yaml`` followed by any
number of ``dotted.key=value`` overrides, so an experiment is fully described
by the YAML file plus the command line that is stored with its results.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml

DEFAULTS: dict[str, Any] = {
    "name": "default",
    "seed": 0,
    "device": "auto",
    "threads": 0,  # CPU threads for PyTorch; 0 keeps the library default
    "out_dir": "runs/default",
    "backbone": {
        # "tiny-random" builds a small model with no download (tests only);
        # otherwise a Hugging Face causal-LM id. The backbone is always frozen.
        "name": "tiny-random",
        # lm_steps > 0 trains the tiny model as a language model first (on the
        # toy corpora, or on the real corpora named in "corpora"); lm_focus /
        # lm_other_share make it a good prior for one corpus and a poorer one
        # for the others, as a general LLM is for a specialised domain
        "tiny": {"hidden": 96, "layers": 2, "heads": 4, "vocab": 600, "lm_steps": 0, "lm_lr": 3.0e-3, "lm_focus": "", "lm_other_share": 1.0, "corpora": []},
    },
    "codec": {
        "granularities": [2, 4, 8, 16],  # real channel symbols per token
        "max_tokens": 48,
        "codebook_hidden": 256,
        "peak_ratio": 4.0,  # per-codeword power limit relative to the average (6 dB)
        # "slots": number of LoRA adapters the receiver can hold (one per knowledge domain)
        "lora": {"r": 8, "alpha": 16, "dropout": 0.0, "targets": ["q_proj", "v_proj"], "last_n": 0, "slots": 2},
    },
    # fast memory at the receiver: the messages of the same conversation that
    # it has already decoded are put in front of the message being decoded
    "context": {"enabled": True, "max_tokens": 64},
    # optional second fast memory: decayed counts of recently decoded tokens
    # mixed into the prior ("weight" is the initial share; the medium level
    # may learn to trust it more or less). Off by default: with a pretrained
    # LLM it was worth almost nothing (see docs/METHOD.md).
    "cache": {"enabled": False, "decay": 0.98, "weight": 0.05, "min_mass": 50.0},
    "channel": {"type": "awgn", "snr_train": [0.0, 20.0]},
    "data": {
        "cache_dir": "data",
        "base": ["europarl"],
        "min_words": 4,
        "max_words": 30,
        "max_sentences": 60000,
    },
    "pretrain": {  # trains the semantic codebook only; no LLM pass is needed
        "steps": 2000,
        "batch_tokens": 256,
        "grans_per_step": 3,  # prefix lengths drawn per step (stochastic nesting objective)
        # share of each batch drawn uniformly from the whole vocabulary rather
        # than from the base corpus, so that tokens the base corpus never
        # uses (the vocabulary of future domains) are separated as well
        "uniform_share": 0.25,
        "lr": 2.0e-3,
        "warmup": 100,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "eval_every": 500,
        "eval_tokens": 2000,
        "log_every": 50,
    },
    "stream": {
        "kind": "hier",  # "hier" (domain/topic regimes) or "temporal"
        "domains": ["ag_news", "teleqna"],
        "cycles": 2,
        "tau_slow": 300,
        "tau_med": 60,
        "kappa": 0.9,
        "temporal_corpus": "huffpost",
        "temporal_steps": 1500,
        "temporal_eras": 4,
        "batch_size": 16,
        "snr": {"mean": 10.0, "rho": 0.95, "sigma": 4.0, "min": 0.0, "max": 20.0},
        "m_op": 4,  # reference fixed prefix length (probes, fixed-rate metrics)
        "extra_grans": 1,  # further prefix lengths in the adaptation loss of each step
        # selective refinement on the stream: a token stops at the first shell
        # at which the receiver's confidence reaches the threshold
        "thresholds": [0.5, 0.7, 0.9, 0.95, 0.99],
        "threshold": 0.9,  # the operating threshold (must be one of "thresholds")
        "target_acc": 0.9,  # token accuracy at which methods are compared in symbols
        "probe_size": 128,
        "gen_every": 50,
        "gen_sentences": 16,
        # every "live_every" steps the received symbols are also decoded for real
        # (free-running selective refinement at every threshold), next to the
        # teacher-forced emulation that scores every step; 0 switches the audit off
        "live_every": 0,
        "live_thresholds": None,  # thresholds decoded in the audit (None: all of "thresholds")
    },
    "nested": {
        # the fast level is the context (every message, no gradient, no sync)
        "periods": {"medium": 10, "slow": 50},  # messages between two shipments to the receiver
        "lrs": {"medium": 1.0e-3, "slow": 3.0e-4},
        # "local": the transmitter learns from every message and ships the
        # accumulated change once per period; "chunk": gradients are
        # accumulated and applied once per period (the literal chunked rule)
        "rule": "local",
        # slow-memory slots: every "select_every" messages the transmitter
        # measures which adapter (or none) predicts the current text best and
        # tells the receiver to switch when another one wins by "select_margin"
        # bits/token
        "select_every": 10,
        "select_margin": 0.05,
        "ema": {"enabled": False, "decay": 0.99, "beta": 0.9},  # optional pull of the slow level towards its running average
        "sync": {"bits": 8, "topk": 1.0},
        "grad_clip": 1.0,
        "adapt_codebook": False,  # also adapt (and synchronise) the codebook at the slow period
        "codebook_lr": 1.0e-4,
        "single_period": "auto",  # matched sync budget for the single-rate baseline
        "ewc_lambda": 10.0,
        "ewc_batches": 20,
    },
    # The receiver's real (free-running) decoder, used wherever symbols are
    # decoded for real on the streams (the audit) - for selective refinement and
    # for the fixed-prefix reference alike. beam: width of the beam search (1:
    # greedy); start_m: the shell sent in the first round of refinement (null:
    # the shortest); reopen: closed tokens may be requested again. Chosen on
    # validation data by scripts/tune_decoder.py (docs/METHOD.md, section 2.5).
    # refine_power: power of a refinement round. "message": every shell keeps the gain computed over the
    # whole message (the runs of versions 0.2-0.4); "subset": a round is scaled to unit power over the
    # tokens it carries, so the power constraint holds in every round (from version 0.5)
    "decoder": {"beam": 1, "start_m": None, "reopen": False, "refine_power": "message"},
    "link": {"traffic_per_step": 1000},
    # Margins of the equivalence tests ("no meaningful difference"), fixed here
    # before the confirmatory runs: two methods are called equivalent when the
    # 90% block-bootstrap interval of their paired per-step difference lies
    # inside (-margin, +margin). See docs/METHOD.md, section 9.
    "stats": {"equiv_margin_bits": 0.05, "equiv_margin_symbols": 0.05, "min_seeds_for_test": 6},
    "eval": {
        "snrs": [0.0, 5.0, 10.0, 15.0, 20.0],
        "channels": ["awgn", "rayleigh"],
        "priors": ["llm_ctx", "llm", "unigram", "none"],  # llm_ctx: the LLM with the true earlier sentences as context
        "n_test": 500,
        "batch_size": 16,
        "sim_model": "sentence-transformers/all-MiniLM-L6-v2",
    },
    # sentences for calibration, the sweep and the live decode; a token is
    # re-requested when its calibrated probability of being right is below
    # the threshold
    "refine": {"n_val": 300, "n_test": 500, "n_live": 100, "live_thresholds": [0.5, 0.7, 0.9]},
}


class Cfg(dict):
    """A dict with attribute access; nested dicts are wrapped on the fly."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:  # pragma: no cover - programming error
            raise AttributeError(key) from exc
        return Cfg(value) if isinstance(value, dict) and not isinstance(value, Cfg) else value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self))


def _deep_update(base: dict, extra: dict) -> dict:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _wrap(node: Any) -> Any:
    if isinstance(node, dict):
        return Cfg({k: _wrap(v) for k, v in node.items()})
    return node


def _set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            raise KeyError(f"unknown config section '{key}' in override '{dotted}'")
        node = node[key]
    if keys[-1] not in node:
        raise KeyError(f"unknown config key '{dotted}'")
    node[keys[-1]] = value


def _check_keys(node: dict, reference: dict, where: str, prefix: str = "") -> None:
    """A config file written for another version of the code fails loudly."""
    for key, value in node.items():
        if key not in reference:
            raise KeyError(f"unknown config key '{prefix}{key}' in {where}")
        if isinstance(value, dict) and isinstance(reference[key], dict):
            _check_keys(value, reference[key], where, f"{prefix}{key}.")


def load_config(path: str | Path | None, overrides: Iterable[str] = ()) -> Cfg:
    """Load DEFAULTS, overlay the YAML file, then apply ``a.b=c`` overrides."""
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        with open(path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        _check_keys(loaded, DEFAULTS, str(path))
        _deep_update(cfg, loaded)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override '{item}' is not of the form key=value")
        key, raw = item.split("=", 1)
        _set_dotted(cfg, key.strip(), yaml.safe_load(raw))
    validate(cfg)
    return _wrap(cfg)


def validate(cfg: dict) -> None:
    grans = cfg["codec"]["granularities"]
    if sorted(set(grans)) != list(grans):
        raise ValueError("codec.granularities must be strictly increasing")
    if cfg["stream"]["m_op"] not in grans:
        raise ValueError("stream.m_op must be one of codec.granularities")
    if cfg["stream"]["threshold"] not in cfg["stream"]["thresholds"]:
        raise ValueError("stream.threshold must be one of stream.thresholds")
    start = cfg["decoder"]["start_m"]
    if start is not None and start not in grans:
        raise ValueError("decoder.start_m must be one of codec.granularities (or null)")
    if int(cfg["decoder"]["beam"]) < 1:
        raise ValueError("decoder.beam must be at least 1")
    if cfg["decoder"]["refine_power"] not in ("message", "subset"):
        raise ValueError("decoder.refine_power must be 'message' or 'subset'")
    audited = cfg["stream"]["live_thresholds"]
    if audited is not None and (not set(audited) <= set(cfg["stream"]["thresholds"]) or cfg["stream"]["threshold"] not in audited):
        raise ValueError("stream.live_thresholds must be a subset of stream.thresholds that contains stream.threshold")
    periods = cfg["nested"]["periods"]
    if not 1 <= periods["medium"] <= periods["slow"]:
        raise ValueError("nested.periods must satisfy 1 <= medium <= slow")
    if cfg["nested"]["rule"] not in ("local", "chunk"):
        raise ValueError("nested.rule must be 'local' or 'chunk'")
    unknown = set(cfg["eval"]["priors"]) - {"llm_ctx", "llm", "unigram", "none"}
    if unknown:
        raise ValueError(f"eval.priors contains unknown entries: {sorted(unknown)}")
    if cfg["channel"]["type"] not in ("awgn", "rayleigh"):
        raise ValueError("channel.type must be 'awgn' or 'rayleigh'")
