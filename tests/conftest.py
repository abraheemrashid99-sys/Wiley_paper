from __future__ import annotations

import pytest
import torch

from dnsc.codec import build_codec, save_codec
from dnsc.config import load_config

TINY_OVERRIDES = [
    "device=cpu",
    "backbone.name=tiny-random",
    "backbone.train_backbone=true",
    "backbone.tiny={hidden: 32, layers: 1, heads: 2, vocab: 400}",
    "codec.n_slots=2",
    "codec.n_prompts=2",
    "codec.granularities=[4, 8, 16]",
    "codec.max_tokens=20",
    "codec.lora={r: 2, alpha: 4, dropout: 0.0, targets: [q_proj, v_proj], tx_last_n: 0, rx_last_n: 0}",
    "codec.deproj_rank=2",
    "data.base=[toy_a]",
    "data.max_sentences=2000",
    "pretrain.steps=6",
    "pretrain.batch_size=4",
    "pretrain.eval_every=3",
    "pretrain.eval_sentences=8",
    "pretrain.log_every=3",
    "pretrain.warmup=2",
    "stream.domains=[toy_b, toy_c]",
    "stream.cycles=1",
    "stream.tau_slow=6",
    "stream.tau_med=3",
    "stream.temporal_corpus=toy_b",
    "stream.temporal_steps=8",
    "stream.temporal_eras=2",
    "stream.batch_size=4",
    "stream.m_op=8",
    "stream.probe_size=6",
    "stream.gen_every=4",
    "stream.gen_sentences=2",
    "nested.periods={fast: 1, medium: 2, slow: 3}",
    "nested.ewc_batches=2",
    "eval.snrs=[0.0, 10.0]",
    "eval.channels=[awgn, rayleigh]",
    "eval.n_test=8",
    "eval.batch_size=4",
    "eval.sim_model=null",
    "refine={quality: bleu1, gamma: 0.5, n_val: 12, n_test: 12}",
]


@pytest.fixture()
def cfg(tmp_path):
    return load_config(None, TINY_OVERRIDES + [f"out_dir={tmp_path / 'run'}", f"data.cache_dir={tmp_path / 'data'}"])


@pytest.fixture()
def codec(cfg):
    torch.manual_seed(0)
    model = build_codec(cfg, torch.device("cpu"))
    model.eval()
    return model


@pytest.fixture()
def checkpoint(cfg, codec, tmp_path):
    directory = tmp_path / "run" / "base"
    save_codec(codec, directory)
    return directory
