"""Codec, decoder and end-to-end integration tests on the tiny offline backbone."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from dnsc.channel import Channel, shell_slices
from dnsc.classical import lm_bits
from dnsc.codec import GROUPS, TokenBatch, TokenCache, join_context, load_codec, slot_group
from dnsc.data import get_corpus
from dnsc.lora import LoRALinear, inject_lora
from dnsc.online import METHODS, SlotSelector, build_levels, matched_period, run_stream
from dnsc.headroom import format_rows, run_headroom
from dnsc.refine import Calibrator, _grid_features, collect, level_stats, run_refinement, selective_decode
from dnsc.train import evaluate_gen, evaluate_tf, pretrain

TEXTS = ["The council adopts the budget on trade today.", "The striker scores a late goal in extra time.", "The scheduler allocates resource blocks in every slot."]
ROOT = Path(__file__).resolve().parents[1]
CPU = torch.device("cpu")




# -------------------------------------------------------------------- LoRA
def test_lora_is_identity_at_init():
    torch.manual_seed(0)
    base = torch.nn.Linear(6, 5)
    layer = LoRALinear(base, r=2, alpha=4, dropout=0.0, sides=("s0", "s1"))
    x = torch.randn(3, 6)
    layer.active = "s0"
    assert torch.equal(layer(x), base(x))  # B is zero-initialised
    with torch.no_grad():
        layer.lora_B["s0"].fill_(0.3)
    assert not torch.allclose(layer(x), base(x))
    layer.active = "s1"  # the other slot is untouched
    assert torch.equal(layer(x), base(x))
    layer.active = None
    assert torch.equal(layer(x), base(x))


def test_lora_wraps_gpt2_style_conv1d():
    from transformers.pytorch_utils import Conv1D

    torch.manual_seed(0)
    base = Conv1D(5, 6)  # 6 inputs, 5 outputs, weight stored (in, out)
    layer = LoRALinear(base, r=2, alpha=4, dropout=0.0, sides=("s0",))
    assert layer.lora_A["s0"].shape == (2, 6) and layer.lora_B["s0"].shape == (5, 2)
    x = torch.randn(3, 6)
    layer.active = "s0"
    assert torch.equal(layer(x), base(x))
    with torch.no_grad():
        layer.lora_B["s0"].fill_(0.3)
    assert layer(x).shape == (3, 5) and not torch.allclose(layer(x), base(x))


def test_lora_last_n_restricts_layers():
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(vocab_size=50, hidden_size=16, intermediate_size=32, num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2)
    assert len(inject_lora(LlamaForCausalLM(config), ["q_proj", "v_proj"], r=2, alpha=4, dropout=0.0)) == 6
    assert len(inject_lora(LlamaForCausalLM(config), ["q_proj", "v_proj"], r=2, alpha=4, dropout=0.0, last_n=1)) == 2
    with pytest.raises(RuntimeError):
        inject_lora(LlamaForCausalLM(config), ["no_such_layer"], r=2, alpha=4, dropout=0.0)


# ------------------------------------------------------------ codebook, power
def test_groups_partition_every_trainable_parameter(codec):
    groups = codec.groups()
    assert codec.n_slots == 2 and set(groups) == set(GROUPS) | {slot_group(1)} and all(groups.values())
    assert slot_group(0) == "rx_lora" and slot_group(1) == "rx_lora#1"
    names = [n for g in groups.values() for n, _ in g]
    assert len(names) == len(set(names))
    assert set(names) == {n for n, _ in codec.named_parameters() if not n.startswith("lm.") or "lora_" in n}
    assert not any(p.requires_grad for n, p in codec.named_parameters() if n.startswith("lm.") and "lora_" not in n)
    assert {n for n, _ in groups["rx_bias"]} == {"prior_weight", "lik_weight", "bias_u", "cache_logit"}
    assert all(n.endswith(".s0") for n, _ in groups["rx_lora"]) and all(n.endswith(".s1") for n, _ in groups["rx_lora#1"])
    assert sum(p.numel() for _, p in groups["rx_lora"]) == sum(p.numel() for _, p in groups["rx_lora#1"])


def test_slots_are_independent_adapters(codec):
    batch = codec.tokenize(TEXTS)
    with torch.no_grad():
        base = codec.lm_logits_tf(batch)
        for name, param in codec.groups()["rx_lora#1"]:
            if "lora_B" in name:
                param.fill_(0.2)
        codec.set_slot(0)
        assert torch.allclose(codec.lm_logits_tf(batch), base, atol=1e-5)  # slot 0 was not touched
        codec.set_slot(1)
        assert not torch.allclose(codec.lm_logits_tf(batch), base, atol=1e-4)
        codec.set_slot(None)
        assert torch.allclose(codec.lm_logits_tf(batch), base, atol=1e-5)  # every adapter off
    with pytest.raises(ValueError):
        codec.set_slot(2)


def test_codebook_meets_the_nested_power_constraint(cfg, checkpoint):
    codec = load_codec(cfg, checkpoint, CPU)
    code = codec.codebook()
    assert code.shape == (codec.vocab, codec.M) and not code.requires_grad
    assert codec.codebook() is code  # cached
    p = codec.log_unigram.exp().unsqueeze(1)
    assert float(p.sum()) == pytest.approx(1.0, abs=1e-4)
    for m in codec.grans:  # average power under the base distribution, at every prefix
        assert 0.9 <= float((p * code[:, :m] ** 2).sum()) / m <= 1.0 + 1e-4
    for sl in shell_slices(codec.grans):
        assert float((code[:, sl] ** 2).sum(1).max()) <= codec.peak * (sl.stop - sl.start) * (1 + 1e-4)
    assert len(torch.unique(code, dim=0)) == codec.vocab  # no two tokens share a codeword


def test_block_power_control_is_exact_for_any_text(cfg, checkpoint):
    """Whatever a message contains, every transmitted shell has unit average
    power, and the reported noise precision matches the simulated noise."""
    codec = load_codec(cfg, checkpoint, CPU)
    texts = get_corpus("toy_c", cfg["data"]).texts("test")[:40] + TEXTS  # out-of-distribution text
    batch = codec.tokenize(texts)
    x = codec.encode(batch)
    assert float(x[~batch.mask].abs().sum()) == 0.0
    gains = codec.block_gains(x, batch.lens)
    sent = x * gains.unsqueeze(1)
    for sl in shell_slices(codec.grans):
        power = (sent[:, :, sl] ** 2).sum(dim=(1, 2)) / (batch.lens * (sl.stop - sl.start))
        assert torch.allclose(power, torch.ones(len(texts)), atol=1e-4)
    for m in codec.grans:  # hence every prefix of every message too
        power = (sent[:, :, :m] ** 2).sum(dim=(1, 2)) / (batch.lens * m)
        assert torch.allclose(power, torch.ones(len(texts)), atol=1e-4)
    assert codec.codebook_power(batch, codec.M) > 0.0

    long_batch = TokenBatch(batch.ids[:1].repeat(1, 400), batch.lens[:1] * 0 + batch.ids.shape[1] * 400)
    xl = codec.codebook()[long_batch.ids]
    y, prec = codec.transmit(xl, long_batch.lens, 7.0, Channel("awgn", 3))
    empirical = ((y - xl) ** 2).mean(dim=1)[0]  # noise variance per symbol position
    assert torch.allclose(empirical * prec[0], torch.ones(codec.M), atol=0.15)
    expected = codec.block_gains(xl, long_batch.lens)[0] ** 2 * 10 ** 0.7
    assert torch.allclose(prec[0], expected, rtol=1e-4)


# ---------------------------------------------------------------- likelihood
def test_likelihood_matches_brute_force(codec):
    torch.manual_seed(1)
    batch = codec.tokenize(TEXTS)
    x = codec.encode(batch)
    y, prec = codec.transmit(x, batch.lens, 5.0, Channel("rayleigh", 0))
    code = codec.codebook()
    for m in codec.grans:
        fast = codec.loglik(y, m, prec)
        assert fast.shape == (3, batch.ids.shape[1], codec.vocab)

        def brute(b, t, v):
            return -0.5 * float((prec[b, :m] * (y[b, t, :m] - code[v, :m]) ** 2).sum())

        for b, t, v in [(0, 0, 3), (1, 2, 17), (2, 1, 250)]:  # equal up to a term that is the same for every codeword
            assert float(fast[b, t, v] - fast[b, t, 0]) == pytest.approx(brute(b, t, v) - brute(b, t, 0), rel=1e-4, abs=1e-3)
        per_token = torch.full_like(batch.ids, m)
        assert torch.allclose(codec.loglik(y, per_token, prec), fast, atol=1e-3)
    mixed = torch.tensor(codec.grans)[torch.arange(batch.ids.numel()).reshape(batch.ids.shape) % len(codec.grans)]
    general = codec.loglik(y, mixed, prec)
    for m in codec.grans:  # each token is scored with its own prefix length
        assert torch.allclose(general[mixed == m], codec.loglik(y, m, prec)[mixed == m], atol=1e-3)
    keep = (torch.arange(codec.M).view(1, 1, -1) < mixed.unsqueeze(-1)).float()
    assert torch.allclose(codec.loglik_w(y, prec.unsqueeze(1) * keep), general, atol=1e-3)


def test_clean_channel_is_decoded_exactly_without_any_prior(cfg, checkpoint):
    codec = load_codec(cfg, checkpoint, CPU)
    texts = get_corpus("toy_a", cfg["data"]).texts("test")[:16]
    for kind in ("awgn", "rayleigh"):
        out = evaluate_gen(codec, texts, codec.M, 60.0, Channel(kind, 0), prior="none", return_texts=True)
        assert out["gen_tok_acc"] == 1.0 and out["exact"] == 1.0 and out["hyps"] == out["refs"]
    ref = codec.reference_texts(["  The council adopts the budget. "])
    assert ref == ["The council adopts the budget."] and codec.reference_texts(ref) == ref


# --------------------------------------------------------------------- prior
def test_prior_is_normalised_and_independent_of_padding(codec):
    alone = codec.prior_tf(codec.tokenize([TEXTS[1]]))
    padded = codec.prior_tf(codec.tokenize([TEXTS[1], TEXTS[0] + " " + TEXTS[2]]))
    width = alone.shape[1]
    assert torch.allclose(alone[0], padded[0, :width], atol=1e-4)
    assert torch.allclose(alone.exp().sum(-1), torch.ones(1, width), atol=1e-4)
    batch = codec.tokenize(TEXTS)
    assert torch.allclose(codec.prior_tf(batch, prior="unigram")[0, 0], codec.log_unigram)
    assert float(codec.prior_tf(batch, prior="none").abs().sum()) == 0.0
    with pytest.raises(ValueError):
        codec.decode_loglik(torch.zeros(1, 2, codec.vocab), torch.tensor([2]), prior="oracle")


def test_free_running_decoder_is_consistent_with_a_full_forward_pass(codec):
    """The cached greedy loop must make exactly the decisions that a full,
    uncached pass over its own output would make."""
    batch = codec.tokenize(TEXTS)
    y, prec = codec.transmit(codec.encode(batch), batch.lens, 3.0, Channel("awgn", 0))
    decoded = codec.decode(y, 2, prec, batch.lens)
    width = batch.ids.shape[1]
    own = torch.zeros_like(batch.ids)
    for i, row in enumerate(decoded.token_ids):
        assert len(row) == int(batch.lens[i])
        own[i, : len(row)] = torch.tensor(row)
    replay = codec.posterior(codec.prior_tf(TokenBatch(own, batch.lens)), codec.loglik(y, 2, prec))
    assert torch.equal(replay.argmax(-1)[batch.mask], own[batch.mask])
    chosen = replay.gather(-1, own.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(chosen[batch.mask], decoded.token_logprob[:, :width][batch.mask], atol=1e-3)
    assert ((decoded.min_prob > 0) & (decoded.min_prob <= 1)).all() and (decoded.mean_entropy >= 0).all()
    assert float(decoded.token_logprob[~batch.mask].abs().sum()) == 0.0


def test_running_text_tokenisation_and_history(codec):
    alone = codec.tokenize(TEXTS[:2])
    running = codec.tokenize(TEXTS[:2], [False, True])
    assert torch.equal(alone.ids[0, : int(alone.lens[0])], running.ids[0, : int(running.lens[0])])  # opens a document: unchanged
    first, later = int(alone.ids[1, 0]), int(running.ids[1, 0])
    assert first != later and codec.tok.decode([later]).startswith(" ")  # continues one: tokenised with its leading space
    assert codec.reference_texts(TEXTS[:2], [False, True]) == codec.reference_texts(TEXTS[:2])  # the text itself is the same
    history = codec.history_ids(TEXTS[:2], 1000)
    assert codec.tok.decode(history) == TEXTS[0] + " " + TEXTS[1]
    second = running.ids[1, : int(running.lens[1])].tolist()
    assert codec.history_ids(TEXTS[:2], len(second)) == second  # only whole sentences are kept ...
    assert codec.history_ids(TEXTS[:2], len(second) - 1) == second[1:]  # ... unless the last one alone is too long
    assert codec.history_ids([], 10) == [] and codec.history_ids(TEXTS, 0) == []
    assert join_context([[1, 2], [3], [4, 5, 6]], 4) == [3, 4, 5, 6] and join_context([[1, 2], [3]], 10) == [1, 2, 3] and join_context([], 5) == []


def test_context_is_read_and_left_padding_changes_nothing(codec):
    """A message decoded with its history in a padded batch must get exactly
    the prior it gets alone; a message without history must be unaffected by
    the histories of its neighbours."""
    flags = [True, False, True]
    batch = codec.tokenize(TEXTS, flags)
    contexts = [codec.history_ids([TEXTS[1], TEXTS[2]], 100), [], codec.history_ids([TEXTS[0]], 100)]
    assert len(contexts[0]) != len(contexts[2])
    with torch.no_grad():
        plain = codec.prior_tf(batch)
        together = codec.prior_tf(batch, contexts=contexts)
        assert torch.allclose(together.exp().sum(-1), torch.ones(together.shape[:2]), atol=1e-4)
        assert torch.allclose(together[1][batch.mask[1]], plain[1][batch.mask[1]], atol=1e-4)  # no history: unchanged
        assert not torch.allclose(together[0][batch.mask[0]], plain[0][batch.mask[0]], atol=1e-3)  # history is read
        for row in (0, 2):
            single = codec.tokenize([TEXTS[row]], [flags[row]])
            alone = codec.prior_tf(single, contexts=[contexts[row]])
            assert torch.allclose(alone[0], together[row, : alone.shape[1]], atol=1e-4)
        flat = codec.prior_tf(batch, contexts=contexts, flat=True)
        assert torch.allclose(flat, together[batch.mask], atol=1e-5)
        assert torch.equal(codec.prior_tf(batch, contexts=[[], [], []]), plain)


def test_free_running_decoder_with_context_matches_a_full_forward_pass(codec):
    flags = [True, False, True]
    batch = codec.tokenize(TEXTS, flags)
    contexts = [codec.history_ids([TEXTS[1], TEXTS[2]], 100), [], codec.history_ids([TEXTS[0]], 100)]
    y, prec = codec.transmit(codec.encode(batch), batch.lens, 3.0, Channel("awgn", 0))
    decoded = codec.decode(y, 2, prec, batch.lens, contexts=contexts)
    own = torch.zeros_like(batch.ids)
    for i, row in enumerate(decoded.token_ids):
        own[i, : len(row)] = torch.tensor(row)
    with torch.no_grad():
        replay = codec.posterior(codec.prior_tf(TokenBatch(own, batch.lens), contexts=contexts), codec.loglik(y, 2, prec))
    assert torch.equal(replay.argmax(-1)[batch.mask], own[batch.mask])
    chosen = replay.gather(-1, own.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(chosen[batch.mask], decoded.token_logprob[:, : batch.ids.shape[1]][batch.mask], atol=1e-3)
    without = codec.decode(y, 2, prec, batch.lens)
    assert without.token_ids[1] == decoded.token_ids[1]  # the row without history decodes as before


def test_the_language_prior_and_the_context_help_at_low_rate(cfg, checkpoint):
    codec = load_codec(cfg, checkpoint, CPU)
    corpus = get_corpus("toy_a", cfg["data"])
    texts = corpus.texts("test")[:48]
    acc = {prior: evaluate_tf(codec, texts, 1, 5.0, Channel("awgn", 1), prior=prior)["tok_acc"] for prior in ("llm", "unigram", "none")}
    assert acc["llm"] > acc["unigram"] + 0.2 and acc["unigram"] >= acc["none"] - 0.02
    gen = {prior: evaluate_gen(codec, texts, 1, 5.0, Channel("awgn", 1), prior=prior)["gen_tok_acc"] for prior in ("llm", "none")}
    assert gen["llm"] > gen["none"]
    by_rate = [evaluate_tf(codec, texts, m, 5.0, Channel("awgn", 2), prior="none")["tok_acc"] for m in codec.grans]
    assert by_rate == sorted(by_rate) and by_rate[-1] > by_rate[0] + 0.2  # more symbols, fewer errors
    # sentences with a history: reading it must make them more predictable
    pairs = [(doc[i]["text"], [r["text"] for r in doc[:i]]) for doc in corpus.documents("test") for i in range(1, len(doc))][:48]
    later = [text for text, _ in pairs]
    contexts = [codec.history_ids(history, 64) for _, history in pairs]
    alone = evaluate_tf(codec, later, 1, 5.0, Channel("awgn", 3))
    informed = evaluate_tf(codec, later, 1, 5.0, Channel("awgn", 3), contexts=contexts, continued=[True] * len(later))
    assert informed["prior_bits"] < alone["prior_bits"] - 0.2 and informed["tok_acc"] > alone["tok_acc"]
    informed_gen = evaluate_gen(codec, later, 1, 5.0, Channel("awgn", 3), contexts=contexts, continued=[True] * len(later))
    assert informed_gen["gen_tok_acc"] > evaluate_gen(codec, later, 1, 5.0, Channel("awgn", 3))["gen_tok_acc"]


def test_classical_reference_bits(codec):
    bits = lm_bits(codec, TEXTS)
    assert bits.shape == (3,) and (bits > 0).all()
    longer = lm_bits(codec, [TEXTS[0] + " " + TEXTS[1]])
    assert longer[0] > bits[0]


# --------------------------------------------------------------------- cache
def test_token_cache(codec):
    cache = TokenCache(codec.vocab, decay=0.5, min_mass=4.0)
    prior = codec.prior_tf(codec.tokenize(TEXTS[:1]))
    weight = torch.sigmoid(codec.cache_logit)
    assert cache.mix(prior, weight) is prior  # an empty cache changes nothing
    cache.update(torch.tensor([7, 7, 7, 9]))
    mixed = cache.mix(prior, weight)
    assert torch.allclose(mixed.exp().sum(-1), torch.ones(prior.shape[:2]), atol=1e-4)
    unseen = torch.ones(codec.vocab, dtype=torch.bool)
    unseen[[7, 9]] = False  # probability moves from the tokens not in the cache to those in it
    assert (mixed[..., 7] > prior[..., 7]).all() and (mixed[..., unseen].exp().sum(-1) < prior[..., unseen].exp().sum(-1)).all()
    cache.update(torch.tensor([], dtype=torch.long))
    assert float(cache.counts[7]) == pytest.approx(1.5) and float(cache.counts.sum()) == pytest.approx(2.0)
    codec.cache_logit.requires_grad_(True)
    cache.mix(prior, torch.sigmoid(codec.cache_logit))[..., 7].sum().backward()
    assert codec.cache_logit.grad is not None and float(codec.cache_logit.grad) > 0


# ---------------------------------------------------------------- checkpoints
def test_pretraining_trains_the_codebook_and_is_skipped_when_done(cfg, checkpoint):
    lines = [json.loads(line) for line in (checkpoint / "pretrain_log.jsonl").read_text().splitlines()]
    losses = [row["loss"] for row in lines if "loss" in row]
    assert losses[-1] < losses[0]
    evals = [row["eval"] for row in lines if "eval" in row]
    assert evals[-1]["4"] > evals[-1]["1"]  # more symbols decode better
    stamp = (checkpoint / "codec.pt").stat().st_mtime_ns
    pretrain(cfg)
    assert (checkpoint / "codec.pt").stat().st_mtime_ns == stamp
    pretrain(cfg, force=True)
    assert (checkpoint / "codec.pt").stat().st_mtime_ns != stamp


def test_checkpoint_round_trip(cfg, checkpoint):
    a = load_codec(cfg, checkpoint, CPU)
    b = load_codec(cfg, checkpoint, CPU)
    assert torch.equal(a.codebook(), b.codebook()) and torch.equal(a.log_unigram, b.log_unigram)
    assert float(a.log_unigram.exp().max()) > 5.0 / a.vocab  # the unigram prior was estimated, not left uniform
    ra = evaluate_tf(a, TEXTS, 2, 10.0, Channel("awgn", 4))
    rb = evaluate_tf(b, TEXTS, 2, 10.0, Channel("awgn", 4))
    assert ra["ce_bits"] == pytest.approx(rb["ce_bits"], abs=1e-5)
    with pytest.raises(FileNotFoundError):
        load_codec(cfg, checkpoint.parent / "missing", CPU)


# ------------------------------------------------------------------- streams
def test_level_schedules(cfg, codec):
    ncfg = cfg["nested"]
    assert build_levels(codec, "frozen", ncfg) == [] and build_levels(codec, "context_only", ncfg) == []
    nested = build_levels(codec, "nested", ncfg)
    assert [(lv.name, lv.side, lv.period, lv.ema) for lv in nested] == [("rx_bias", "rx", 2, False), ("rx_lora", "rx", 3, False), ("rx_lora#1", "rx", 3, False)]
    assert [lv.name for lv in build_levels(codec, "nested_no_slots", ncfg)] == ["rx_bias", "rx_lora"]
    assert [lv.name for lv in build_levels(codec, "nested_no_context", ncfg)] == ["rx_bias", "rx_lora", "rx_lora#1"]
    assert {lv.period for lv in build_levels(codec, "full_online", ncfg)} == {1}
    single = build_levels(codec, "single_rate", ncfg)
    period = matched_period(codec, ncfg)
    assert [lv.name for lv in single] == ["rx_bias", "rx_lora"] and {lv.period for lv in single} == {period} and 2 <= period <= 3
    assert {lv.period for lv in build_levels(codec, "nested_two_level", ncfg)} == {period}

    def rate(levels):  # sync bits per step; one slot is active at a time
        return sum((32 * len(lv.params) + 8 * lv.n_params) / lv.period for lv in levels if "#" not in lv.name)

    assert rate(single) == pytest.approx(rate(nested), rel=0.35)  # matched budget up to period rounding
    ncfg["ema"]["enabled"] = True
    assert [lv.ema for lv in build_levels(codec, "nested", ncfg)] == [False, True, True]
    ncfg["adapt_codebook"] = True
    assert [lv.name for lv in build_levels(codec, "nested_no_slots", ncfg)] == ["rx_bias", "rx_lora", "codebook"]
    with pytest.raises(ValueError):
        build_levels(codec, "not_a_method", ncfg)


def test_slot_selector():
    chooser = SlotSelector(2, margin=0.05)
    assert chooser.active == 0 and chooser.bits == 2 and chooser.trained == [False, False]
    assert chooser.choose({None: 5.0, 0: 5.0, 1: 5.0}) == 0  # nothing is better: keep the slot
    chooser.mark_trained()
    assert chooser.choose({None: 5.0, 0: 4.99, 1: 5.0}) == 0  # within the margin
    # a new domain: the adapter of the old one hurts, the frozen prior is best -> start the unused slot
    assert chooser.choose({None: 7.0, 0: 7.9, 1: 7.0}) == 1 and chooser.switches == 1
    chooser.mark_trained()
    assert chooser.choose({None: 5.0, 0: 4.8, 1: 5.9}) == 0  # the old domain returns: its adapter is recalled
    assert chooser.choose({None: 7.0, 0: 7.9, 1: 6.2}) == 1 and chooser.switches == 3
    # a third domain with no slot left: fall back to the frozen prior rather than a harmful adapter
    assert chooser.choose({None: 6.0, 0: 6.5, 1: 6.4}) is None and chooser.active is None
    chooser.mark_trained()  # nothing to mark while no slot is active
    assert chooser.trained == [True, True]
    assert chooser.choose({None: 6.0, 0: 5.5, 1: 6.4}) == 0


@pytest.mark.parametrize("method", METHODS)
def test_stream_runs_for_every_method(cfg, checkpoint, tmp_path, method):
    out = tmp_path / f"{method}.json"
    result = run_stream(cfg, method, seed=0, ckpt_dir=checkpoint, out_path=out)
    saved = json.loads(out.read_text())
    assert saved["n_steps"] == 12 and len(saved["log"]["tok_acc"]) == 12
    assert len(saved["probes"]) == 3 and set(saved["probes"][0]["domains"]) == {"toy_b", "toy_c"}
    assert len(saved["ledger"]["cumulative"]) == 12 and saved["ledger"]["cumulative"][-1] == saved["summary"]["sync_bits"]
    assert saved["context"] == (method not in ("frozen", "nested_no_context"))
    assert saved["slots"] == {"frozen": 0, "context_only": 0, "nested": 2, "nested_no_context": 2, "nested_two_level": 2}.get(method, 1)
    payload = saved["summary"]["payload_symbols"]
    assert payload == pytest.approx(saved["m_op"] * saved["summary"]["tokens_per_message"])
    if method in ("frozen", "context_only"):
        assert result["summary"]["sync_bits"] == 0 and result["summary"]["net_rate"] == pytest.approx(payload)
    else:
        assert result["summary"]["sync_bits"] > 0 and result["summary"]["net_rate"] > payload
    if saved["slots"] == 2:  # the choice of slot is signalled and charged: 6 selections of 2 bits in 12 steps
        assert saved["ledger"]["bits"]["slot_select"] == 12 and set(saved["log"]["slot"]) <= {-1, 0, 1}
    else:
        assert "slot_select" not in saved["ledger"]["bits"]
    assert np.isfinite(saved["log"]["ce_bits"]).all() and all(0.0 <= a <= 1.0 for a in saved["log"]["tok_acc"])


def test_streams_are_prequential_and_deterministic(cfg, checkpoint, tmp_path):
    a = run_stream(cfg, "frozen", 0, checkpoint, tmp_path / "a.json")
    b = run_stream(cfg, "frozen", 0, checkpoint, tmp_path / "b.json")
    assert a["log"]["ce_bits"] == b["log"]["ce_bits"]  # deterministic given the seed
    assert a["probes"][0]["domains"]["toy_b"]["ce_bits"] == pytest.approx(a["probes"][-1]["domains"]["toy_b"]["ce_bits"], abs=1e-6)
    c = run_stream(cfg, "nested_no_context", 0, checkpoint, tmp_path / "c.json")
    assert c["probes"][0]["domains"]["toy_b"]["ce_bits"] != pytest.approx(c["probes"][-1]["domains"]["toy_b"]["ce_bits"], abs=1e-6)
    assert a["log"]["ce_bits"][0] == pytest.approx(c["log"]["ce_bits"][0], abs=1e-5)  # scored before the first update
    d = run_stream(cfg, "context_only", 0, checkpoint, tmp_path / "d.json")
    assert d["log"]["ce_bits"] != a["log"]["ce_bits"] and d["summary"]["sync_bits"] == 0
    assert d["summary"]["mean_prior_bits"] < a["summary"]["mean_prior_bits"]  # what was decoded before helps
    restored = load_codec(cfg, checkpoint, CPU)
    assert float(restored.bias_u.detach().abs().sum()) == 0.0  # a stream run never writes to the checkpoint


def test_stream_reports_every_prefix_length_and_the_selective_curve(cfg, checkpoint, tmp_path):
    result = run_stream(cfg, "context_only", 0, checkpoint, tmp_path / "r.json")
    grans = cfg["codec"]["granularities"]
    fixed, selective = result["fixed_curve"], result["selective_curve"]
    assert [p["m"] for p in fixed] == grans and [p["threshold"] for p in selective] == cfg["stream"]["thresholds"]
    k_op = grans.index(cfg["stream"]["m_op"])
    assert fixed[k_op]["tok_acc"] == pytest.approx(result["summary"]["mean_tok_acc"])
    assert fixed[k_op]["ce_bits"] == pytest.approx(result["summary"]["mean_ce_bits"])
    assert fixed[-1]["tok_acc"] > fixed[0]["tok_acc"]  # more symbols, fewer errors
    symbols = [p["symbols"] for p in selective]
    assert symbols == sorted(symbols) and grans[0] <= symbols[0] and symbols[-1] <= grans[-1]
    assert selective[-1]["tok_acc"] >= selective[0]["tok_acc"]
    summary = result["summary"]
    assert summary["sel_symbols"] == selective[cfg["stream"]["thresholds"].index(cfg["stream"]["threshold"])]["symbols"]
    assert summary["sel_net_rate"] == pytest.approx(summary["sel_symbols"] * summary["tokens_per_message"])  # no sync cost here
    assert 0.0 <= summary["ece"] <= 1.0 and summary["mean_prior_bits"] > 0
    assert len(result["log"]["acc_by_m"]) == result["n_steps"] and len(result["log"]["sel_symbols"][0]) == len(selective)
    cfg["stream"]["threshold"] = 0.123
    with pytest.raises(ValueError):
        from dnsc.config import validate

        validate(cfg)


def test_level_stats_match_the_per_level_loss(codec):
    batch = codec.tokenize(TEXTS)
    with torch.no_grad():
        y, prec = codec.transmit(codec.encode(batch), batch.lens, 5.0, Channel("awgn", 3))
        log_prior = codec.prior_tf(batch)
        ids, y_flat, w_flat = codec.flat_channel(batch, y, prec)
        flat_prior = codec.prior_tf(batch, flat=True)
        assert torch.allclose(flat_prior, log_prior[batch.mask], atol=1e-5)  # the flat view is the padded one without padding
        stats = level_stats(codec, ids, y_flat, w_flat, flat_prior, entropy=True)
        n_tokens = int(batch.lens.sum())
        assert all(stats[key].shape == (n_tokens, len(codec.grans)) for key in ("correct", "conf", "ce", "decided", "entropy"))
        for k, m in enumerate(codec.grans):
            ce, acc, decided = codec.loss_at(batch, y, prec, m, log_prior)
            lens = batch.lens.float()
            assert stats["ce"][:, k].sum().item() == pytest.approx((ce * lens).sum().item(), rel=1e-4)
            assert stats["correct"][:, k].sum().item() == pytest.approx((acc * lens).sum().item(), rel=1e-4)
            assert torch.equal(stats["decided"][:, k], decided[batch.mask])
            assert codec.flat_loss(ids, y_flat, w_flat, m, flat_prior).item() == pytest.approx(stats["ce"][:, k].mean().item(), rel=1e-4)
    assert ((stats["conf"] > 0) & (stats["conf"] <= 1 + 1e-5)).all() and (stats["entropy"] > -1e-5).all()
    for prior in ("unigram", "none"):
        assert codec.prior_tf(batch, prior=prior, flat=True).shape == (n_tokens, codec.vocab)


def test_headroom_diagnostic(cfg, checkpoint, tmp_path):
    before = load_codec(cfg, checkpoint, CPU).codec_state()
    result = run_headroom(cfg, checkpoint, tmp_path / "h.json", n=12, steps=3)
    assert (tmp_path / "h.json").exists()
    names = ["toy_a", "toy_b", "toy_c"]
    frozen = [row for row in result["rows"] if row["variant"] == "frozen"]
    context = [row for row in result["rows"] if row["variant"] == "context"]
    assert [row["domain"] for row in frozen] == names and all(row["n"] == 12 for row in frozen)
    assert all(row["with_history"] > 0 for row in context)
    assert context[0]["prior_bits"] < frozen[0]["prior_bits"]  # on its own domain the tiny LM uses the history
    adapted = [row for row in result["rows"] if row["variant"] == "context+bias+lora"]
    assert {(row["domain"], row["measured_on"]) for row in adapted} == {(d, o) for d in ("toy_b", "toy_c") for o in names}
    own = next(row for row in adapted if row["domain"] == "toy_b" and row["measured_on"] == "toy_b")
    assert own["prior_bits"] != pytest.approx(context[1]["prior_bits"], abs=1e-6)  # the adaptation changed the prior
    table = format_rows(result)
    assert table and {"measured on", "memory", "Δ prior", "token acc"} <= set(table[0])
    assert any("adapted on toy_c" in row["memory"] for row in table)
    after = load_codec(cfg, checkpoint, CPU).codec_state()
    assert all(torch.equal(before[key], after[key]) for key in before)  # the checkpoint is untouched


def test_receiver_is_scored_with_its_stale_copy(cfg, tmp_path):
    """The transmitter learns from the first message on, but the receiver
    only changes when a level is shipped (medium: every 2 steps)."""
    cfg["codec"]["lora"]["slots"] = 1  # one adapter: no slot is ever switched, so the shipments are regular
    checkpoint = pretrain(cfg)
    frozen = run_stream(cfg, "frozen", 0, checkpoint, tmp_path / "f.json")
    nested = run_stream(cfg, "nested_no_context", 0, checkpoint, tmp_path / "n.json")
    assert nested["rule"] == "local" and nested["slots"] == 1
    assert nested["log"]["ce_bits"][:2] == pytest.approx(frozen["log"]["ce_bits"][:2], abs=1e-5)
    assert nested["log"]["ce_bits"][2] != pytest.approx(frozen["log"]["ce_bits"][2], abs=1e-6)
    assert nested["ledger"]["events"] == {"rx_bias": 6, "rx_lora": 4}  # 12 steps at periods 2 and 3
    cfg["nested"]["rule"] = "chunk"
    chunk = run_stream(cfg, "nested_no_context", 0, checkpoint, tmp_path / "c.json")
    # same shipments; an update that is exactly zero (LoRA "A" before "B" has moved) costs one flag bit
    assert chunk["rule"] == "chunk" and chunk["ledger"]["events"] == nested["ledger"]["events"]
    assert chunk["summary"]["sync_bits"] <= nested["summary"]["sync_bits"]
    assert chunk["log"]["ce_bits"][:2] == pytest.approx(frozen["log"]["ce_bits"][:2], abs=1e-5)


def test_slots_protect_against_negative_transfer(cfg, tmp_path):
    """With an aggressive adapter, what is learned on one kind of text damages
    the prior for another. One adapter is then dragged back and forth; with
    slots, the transmitter moves to another adapter - or back to the frozen
    prior - whenever that predicts the current text better."""
    cfg = copy.deepcopy(cfg)
    cfg["stream"].update({"cycles": 2, "tau_slow": 40, "tau_med": 10, "gen_every": 0, "batch_size": 8})
    cfg["nested"].update({"lrs": {"medium": 5.0e-3, "slow": 1.0e-2}, "select_every": 4, "select_margin": 0.02})
    checkpoint = pretrain(cfg)
    slots = run_stream(cfg, "nested", 0, checkpoint, tmp_path / "s.json")
    single = run_stream(cfg, "nested_no_slots", 0, checkpoint, tmp_path / "o.json")
    used = slots["log"]["slot"]
    assert slots["slot_switches"] >= 2 and {0, 1} <= set(used)  # both adapters were put to work
    assert slots["ledger"]["bits"]["slot_select"] == 40 * 2  # every selection is signalled and paid for
    assert set(single["log"]["slot"]) == {0} and single["slot_switches"] == 0
    returns = slice(80, 160)  # the second visit of both domains
    assert np.mean(slots["log"]["prior_bits"][returns]) < np.mean(single["log"]["prior_bits"][returns])
    assert slots["summary"]["mean_prior_bits"] < single["summary"]["mean_prior_bits"]
    kept = np.mean([v["prior_bits"] for v in slots["probes"][-1]["domains"].values()])
    assert kept < np.mean([v["prior_bits"] for v in single["probes"][-1]["domains"].values()])  # more is retained at the end
    assert all(v["slot"] in (-1, 0, 1) for v in slots["probes"][-1]["domains"].values())


def test_adaptation_helps_on_a_mismatched_domain(cfg, tmp_path):
    """Functional check of the whole adaptation path: when the prior has never
    seen the stream domains, adapting it must clearly beat not adapting, and
    the multi-timescale schedule must cost a fraction of per-message updates."""
    cfg = copy.deepcopy(cfg)
    cfg["backbone"]["tiny"]["lm_other_share"] = 0.0
    cfg["stream"].update({"tau_slow": 60, "tau_med": 15, "gen_every": 0, "batch_size": 8})
    cfg["nested"].update({"lrs": {"medium": 5.0e-3, "slow": 2.0e-3}})
    checkpoint = pretrain(cfg)
    reference = run_stream(cfg, "context_only", 0, checkpoint, tmp_path / "f.json")
    nested = run_stream(cfg, "nested", 0, checkpoint, tmp_path / "n.json")
    online = run_stream(cfg, "full_online", 0, checkpoint, tmp_path / "o.json")
    base = reference["summary"]["mean_ce_bits"]
    assert online["summary"]["mean_ce_bits"] < base - 0.3 and nested["summary"]["mean_ce_bits"] < base - 0.3
    late = slice(60, 120)
    assert np.mean(online["log"]["tok_acc"][late]) > np.mean(reference["log"]["tok_acc"][late])
    assert nested["summary"]["sync_bits"] < 0.5 * online["summary"]["sync_bits"]
    assert nested["summary"]["mean_prior_bits"] < reference["summary"]["mean_prior_bits"] - 0.3


def test_codebook_adaptation_option(cfg, checkpoint, tmp_path):
    cfg["nested"]["adapt_codebook"] = True
    result = run_stream(cfg, "nested", 0, checkpoint, tmp_path / "cb.json")
    assert [lv["name"] for lv in result["levels"]] == ["rx_bias", "rx_lora", "rx_lora#1", "codebook"]
    assert result["ledger"]["bits"]["codebook"] > 0 and np.isfinite(result["log"]["ce_bits"]).all()


def test_temporal_stream_runs(cfg, checkpoint, tmp_path):
    cfg["stream"]["kind"] = "temporal"
    result = run_stream(cfg, "nested", 0, checkpoint, tmp_path / "t.json")
    assert result["n_steps"] == 8 and set(result["probes"][-1]["domains"]) == {"era0", "era1"}


# ---------------------------------------------------------------- refinement
def test_selective_refinement(cfg, checkpoint):
    codec = load_codec(cfg, checkpoint, CPU)
    texts = get_corpus("toy_a", cfg["data"]).texts("test")
    val, test = texts[:24], texts[24:48]
    data = collect(codec, test, 5.0, Channel("awgn", 0), batch_size=8)
    n_tokens = int(codec.tokenize(test).lens.sum())
    assert data["correct"].shape == (n_tokens, len(codec.grans)) and ((data["conf"] > 0) & (data["conf"] <= 1)).all()
    assert data["correct"][:, -1].mean() >= data["correct"][:, 0].mean()

    res, calibrator = run_refinement(codec, val, test, 5.0, "awgn", batch_size=8)
    symbols = [c["symbols"] for c in res["curve"]]
    assert symbols == sorted(symbols) and symbols[0] == codec.grans[0] and symbols[-1] == codec.grans[-1]
    assert res["oracle"]["quality"] >= max(f["quality"] for f in res["fixed"]) - 1e-9
    assert res["oracle"]["symbols"] <= codec.grans[-1] and 0.0 <= res["ece_calibrated"] <= 1.0
    assert all("gain_over_fixed" in c for c in res["curve"])
    assert isinstance(calibrator, Calibrator) and _grid_features(data).shape == (n_tokens * len(codec.grans), 2)

    never = selective_decode(codec, test, 5.0, Channel("awgn", 1), calibrator, threshold=0.0, batch_size=8)
    always = selective_decode(codec, test, 5.0, Channel("awgn", 1), calibrator, threshold=2.0, batch_size=8)
    middle = selective_decode(codec, test, 5.0, Channel("awgn", 1), calibrator, threshold=0.8, batch_size=8)
    assert never["symbols"] == pytest.approx(codec.grans[0]) and never["requests_per_token"] == 0.0
    assert always["symbols"] == pytest.approx(codec.grans[-1]) and always["requests_per_token"] == pytest.approx(len(codec.grans) - 1)
    assert never["symbols"] <= middle["symbols"] <= always["symbols"]
    assert always["gen_tok_acc"] >= never["gen_tok_acc"]


# ---------------------------------------------------- real-backbone code path
def test_pretrained_backbone_path(cfg, tmp_path):
    """The code path used for real backbones: a causal LM and tokenizer loaded
    by name through ``Auto*``, stored in bfloat16 like SmolLM2, kept frozen,
    and *not* written into the codec checkpoint."""
    from transformers import LlamaConfig, LlamaForCausalLM

    from dnsc.backbone import load_tokenizer
    from dnsc.codec import build_codec

    hub = tmp_path / "fake-hub-model"
    tokenizer = load_tokenizer(cfg["backbone"])
    tokenizer.save_pretrained(str(hub))
    torch.manual_seed(1)
    config = LlamaConfig(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    LlamaForCausalLM(config).to(torch.bfloat16).save_pretrained(str(hub))

    cfg["backbone"]["name"] = str(hub)
    codec = build_codec(cfg, CPU)
    assert all(p.dtype == torch.float32 for p in codec.parameters())
    directory = pretrain(cfg)
    state = torch.load(directory / "codec.pt", map_location="cpu")["state"]
    assert state and all(not k.startswith("lm.") or "lora_" in k for k in state)  # backbone is not duplicated on disk
    result = run_stream(cfg, "nested", 0, directory, tmp_path / "s.json")
    assert result["summary"]["sync_bits"] > 0 and np.isfinite(result["log"]["ce_bits"]).all()
    restored = load_codec(cfg, directory, CPU)
    assert evaluate_gen(restored, TEXTS, restored.M, 60.0, Channel("awgn", 0), prior="none")["gen_tok_acc"] == 1.0


# ------------------------------------------------------------------- scripts
def test_tiny_backbone_can_be_trained_on_named_corpora(cfg):
    """The rehearsal path: tokenizer and tiny LM built from corpora of the
    data registry instead of the built-in toy text."""
    from dnsc.codec import build_codec

    cfg["backbone"]["tiny"].update({"corpora": ["toy_a", "toy_b"], "lm_steps": 30, "lm_focus": "toy_a", "lm_other_share": 0.1})
    codec = build_codec(cfg, CPU)
    known = codec.tokenize(["the council adopts the budget on trade"])
    unknown = codec.tokenize(["the base station allocates resource blocks"])  # toy_c was not in the tokenizer's training text
    assert int(known.lens[0]) < int(unknown.lens[0])
    with torch.no_grad():
        assert torch.isfinite(codec.prior_tf(known)).all()


def test_command_line_pipeline(tmp_path):
    """Every script runs end to end from the shell on the offline config."""
    overrides = [f"out_dir={tmp_path / 'run'}", f"data.cache_dir={tmp_path / 'data'}", "pretrain.steps=40", "pretrain.eval_every=20", "stream.tau_slow=8", "stream.cycles=1", "stream.gen_every=4", "eval.n_test=8", "eval.snrs=[10.0]", "refine={n_val: 8, n_test: 8, n_live: 4, live_thresholds: [0.7]}"]

    def run(script: str, *extra: str) -> str:
        cmd = [sys.executable, str(ROOT / "scripts" / script), "--config", str(ROOT / "configs" / "smoke.yaml"), *extra, *overrides]
        done = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT, timeout=900)
        assert done.returncode == 0, f"{script} failed:\n{done.stdout[-2000:]}\n{done.stderr[-2000:]}"
        return done.stdout

    run("check_env.py")
    run("prepare_data.py")
    run("pretrain.py")
    assert "nothing to do" in run("pretrain.py")
    run("eval_rate.py")
    run("run_stream.py", "--methods", "frozen,nested,single_rate", "--seeds", "0,1")
    assert "skip existing" in run("run_stream.py", "--methods", "frozen", "--seeds", "0")
    run("eval_refine.py")
    run("headroom.py", "--n", "8", "--steps", "3")
    run("benchmark_speed.py", "--steps", "2", "--repeats", "2")
    run("aggregate.py")
    run("make_figures.py")
    out = tmp_path / "run"
    expected = ["tables/stream_hier.md", "tables/netrate_hier.csv", "tables/rate_toy_a_bleu4.tex", "tables/rate_toy_a_power.md", "tables/refine_toy_a.md", "tables/refine_toy_a_live.md", "tables/headroom.md", "eval/headroom.json", "figures/stream_hier.png", "figures/rate_toy_a.pdf", "figures/refine_toy_a.png", "eval/speed.json"]
    for name in expected:
        assert (out / name).exists(), name


def test_pipeline_runner_plans_every_stage():
    def run(*args: str) -> str:
        cmd = [sys.executable, str(ROOT / "scripts" / "run_pipeline.py"), "--config", "configs/laptop.yaml", "--dry-run", *args]
        done = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT, timeout=60)
        assert done.returncode == 0, done.stderr
        return done.stdout

    everything = run("--stage", "all", "--seeds", "0,1", "threads=4")
    assert everything.count("=====") == 11 and "pretrain.py --config configs/laptop.yaml threads=4" in everything
    assert "--seeds 0,1 stream.kind=temporal threads=4" in everything
    assert run("--stage", "smoke").count("=====") == 9 and run("--stage", "report").count("=====") == 2
    pilot = run("--stage", "pilot")
    assert pilot.count("=====") == 3 and "--seeds 0 stream.kind=hier" in pilot and "full_online" in pilot


def test_real_configs_are_valid():
    from dnsc.config import load_config

    for name in ("smoke", "rehearsal", "laptop", "main", "scale"):
        loaded = load_config(ROOT / "configs" / f"{name}.yaml")
        assert loaded["stream"]["m_op"] in loaded["codec"]["granularities"], name
