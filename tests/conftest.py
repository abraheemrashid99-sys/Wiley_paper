from __future__ import annotations

import pytest
import torch

from dnsc.codec import build_codec
from dnsc.config import load_config
from dnsc.train import pretrain

TINY_OVERRIDES = [
    "device=cpu",
    "backbone.name=tiny-random",
    "backbone.tiny={hidden: 32, layers: 1, heads: 2, vocab: 400, lm_steps: 400, lm_focus: toy_a, lm_other_share: 0.1}",
    "codec.granularities=[1, 2, 4]",
    "codec.max_tokens=32",
    "codec.codebook_hidden=16",
    "codec.lora={r: 2, alpha: 4, dropout: 0.0, targets: [q_proj, v_proj], last_n: 0, slots: 2}",
    "context={enabled: true, max_tokens: 64}",
    "nested.select_every=2",
    "data.base=[toy_a]",
    "data.max_sentences=2000",
    "pretrain.steps=40",
    "pretrain.batch_tokens=64",
    "pretrain.lr=5.0e-3",
    "pretrain.eval_every=20",
    "pretrain.eval_tokens=200",
    "pretrain.log_every=10",
    "pretrain.warmup=2",
    "stream.domains=[toy_b, toy_c]",
    "stream.cycles=1",
    "stream.tau_slow=6",
    "stream.tau_med=3",
    "stream.temporal_corpus=toy_b",
    "stream.temporal_steps=8",
    "stream.temporal_eras=2",
    "stream.batch_size=4",
    "stream.m_op=2",
    "stream.probe_size=6",
    "stream.gen_every=4",
    "stream.gen_sentences=2",
    "nested.periods={medium: 2, slow: 3}",
    "nested.ewc_batches=2",
    "eval.snrs=[0.0, 10.0]",
    "eval.channels=[awgn, rayleigh]",
    "eval.n_test=8",
    "eval.batch_size=4",
    "eval.sim_model=null",
    "refine={n_val: 12, n_test: 12, n_live: 6, live_thresholds: [0.5, 0.9]}",
]


@pytest.fixture()
def cfg(tmp_path):
    return load_config(None, TINY_OVERRIDES + [f"out_dir={tmp_path / 'run'}", f"data.cache_dir={tmp_path / 'data'}"])


@pytest.fixture()
def codec(cfg):
    """An untrained codec (random codebook) on the warmed-up tiny LM."""
    torch.manual_seed(0)
    model = build_codec(cfg, torch.device("cpu"))
    model.eval()
    return model


@pytest.fixture()
def checkpoint(cfg):
    """A pretrained checkpoint (codebook trained for a few steps)."""
    return pretrain(cfg)
