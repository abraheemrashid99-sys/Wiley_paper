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
    "out_dir": "runs/default",
    "backbone": {
        # "tiny-random" builds a small randomly initialised model with no
        # download (tests only); otherwise a Hugging Face causal-LM id.
        "name": "tiny-random",
        "train_backbone": False,
        "tiny": {"hidden": 96, "layers": 2, "heads": 4, "vocab": 600, "lm_steps": 0},
    },
    "codec": {
        "n_slots": 8,  # transmitter memory tokens
        "n_prompts": 8,  # receiver soft-prompt tokens
        "granularities": [16, 32, 64, 128, 256],  # channel symbols per message
        "max_tokens": 40,
        "lora": {
            "r": 8,
            "alpha": 16,
            "dropout": 0.0,
            "targets": ["q_proj", "v_proj"],
            "tx_last_n": 0,  # 0 = every layer
            "rx_last_n": 0,
        },
        "deproj_rank": 4,
    },
    "channel": {"type": "awgn", "snr_train": [0.0, 20.0]},
    "data": {
        "cache_dir": "data",
        "base": ["europarl"],
        "min_words": 4,
        "max_words": 30,
        "max_sentences": 60000,
    },
    "pretrain": {
        "steps": 6000,
        "batch_size": 16,
        "lr": 1.0e-3,
        "lora_lr": 2.0e-4,
        "warmup": 200,
        "weight_decay": 0.0,
        "grans_per_step": 2,
        "grad_clip": 1.0,
        "eval_every": 500,
        "eval_sentences": 256,
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
        "m_op": 64,
        "extra_grans": 1,
        "probe_size": 128,
        "gen_every": 50,
        "gen_sentences": 16,
    },
    "nested": {
        "periods": {"fast": 1, "medium": 20, "slow": 100},
        "lrs": {"fast": 1.0e-3, "medium": 5.0e-4, "slow": 1.0e-4},
        "ema": {"enabled": True, "decay": 0.99, "beta": 0.9},
        "sync": {"bits": 8, "topk": 1.0},
        "grad_clip": 1.0,
        "single_period": "auto",  # matched sync budget for the single-rate baseline
        "single_lr": 5.0e-4,
        "ewc_lambda": 10.0,
        "ewc_batches": 20,
    },
    "link": {"traffic_per_step": 1000},
    "eval": {
        "snrs": [0.0, 5.0, 10.0, 15.0, 20.0],
        "channels": ["awgn", "rayleigh"],
        "n_test": 500,
        "batch_size": 32,
        "sim_model": "sentence-transformers/all-MiniLM-L6-v2",
    },
    "refine": {"quality": "bleu1", "gamma": 0.8, "n_val": 300, "n_test": 500},
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


def load_config(path: str | Path | None, overrides: Iterable[str] = ()) -> Cfg:
    """Load DEFAULTS, overlay the YAML file, then apply ``a.b=c`` overrides."""
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        with open(path, "r", encoding="utf-8") as handle:
            _deep_update(cfg, yaml.safe_load(handle) or {})
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
    periods = cfg["nested"]["periods"]
    if not periods["fast"] <= periods["medium"] <= periods["slow"]:
        raise ValueError("nested.periods must satisfy fast <= medium <= slow")
    if cfg["channel"]["type"] not in ("awgn", "rayleigh"):
        raise ValueError("channel.type must be 'awgn' or 'rayleigh'")
