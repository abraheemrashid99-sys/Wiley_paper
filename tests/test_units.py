"""Unit tests for the pieces whose correctness the paper's numbers rest on."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from dnsc.channel import Channel, capacity_real, nested_power_normalize, shell_slices
from dnsc.classical import deflate_bits, delivered, max_bits
from dnsc.config import load_config
from dnsc.data import get_corpus
from dnsc.data.corpora import assign_split, process
from dnsc.data.stream import HierStream, TemporalStream, build_probes, snr_trace
from dnsc.data.text import clean, keep_sentence, split_sentences, strip_bracket_tag
from dnsc.linkbudget import crossover_traffic, net_semantic_rate, overhead_fraction, symbols_at_accuracy, sync_symbols
from dnsc.metrics import bleu_scores, expected_calibration_error, exact_match, sentence_bleu
from dnsc.nested import Level, NestedOptimizer, quantize_delta
from dnsc.refine import Calibrator, features, interpolate_fixed, random_matched, reference_policies, stop_levels, stopping_curve
from dnsc.stats import block_bootstrap, equivalence, holm, mean_sd, paired_bootstrap, paired_permutation
from dnsc.train import lr_at, sample_grans


# ------------------------------------------------------------------ config
def test_config_overrides_and_validation():
    cfg = load_config(None, ["seed=7", "nested.sync.bits=4", "codec.granularities=[8, 16]", "stream.m_op=16"])
    assert cfg.seed == 7 and cfg.nested.sync.bits == 4 and cfg["codec"]["granularities"] == [8, 16]
    with pytest.raises(ValueError):
        load_config(None, ["eval.priors=[llm, oracle]"])
    with pytest.raises(KeyError):
        load_config(None, ["nested.not_a_key=1"])
    with pytest.raises(ValueError):
        load_config(None, ["stream.m_op=7"])  # not a granularity
    with pytest.raises(ValueError):
        load_config(None, ["codec.granularities=[16, 8]"])
    with pytest.raises(ValueError):
        load_config(None, ["nested.periods={medium: 20, slow: 10}"])


# -------------------------------------------------------------------- text
def test_sentence_processing():
    assert split_sentences("Stocks fell today. Oil rose sharply! Why?") == ["Stocks fell today.", "Oil rose sharply!", "Why?"]
    assert clean("Short-sellers\\band  cynics #39;win#39; (Reuters)") == "Short-sellers band cynics 'win'"
    assert strip_bracket_tag("What is the purpose of the MFAF? [3GPP Release 18]") == "What is the purpose of the MFAF?"
    assert keep_sentence("The council adopts the budget today.", 4, 30)
    assert not keep_sentence("Too short.", 4, 30)
    assert not keep_sentence("Visit http://example.com for more details today.", 4, 30)
    assert not keep_sentence("12 34 56 78 90 11 22", 4, 30)


def test_splits_are_document_disjoint(cfg):
    corpus = get_corpus("toy_a", cfg["data"])
    docs = {split: {r["doc"] for r in corpus.split(split)} for split in ("train", "val", "test")}
    assert docs["train"] and docs["val"] and docs["test"]
    assert not (docs["train"] & docs["val"]) and not (docs["train"] & docs["test"]) and not (docs["val"] & docs["test"])
    assert assign_split("toy_a", 5) == assign_split("toy_a", 5)
    again = get_corpus("toy_a", cfg["data"])  # served from cache, identical
    assert [r["text"] for r in again.records] == [r["text"] for r in corpus.records]


def test_process_dedups_and_caps():
    raw = [{"text": "The council adopts the budget today. The council adopts the budget today.", "topic": "t", "time": "", "doc": d} for d in range(50)]
    assert len(process("x", raw, 4, 30, 1000)) == 1
    raw = [{"text": f"The council adopts budget number {d} again today.", "topic": "t", "time": "", "doc": d} for d in range(50)]
    assert len(process("x", raw, 4, 30, 10)) == 10


# ----------------------------------------------------------------- channel
def test_nested_power_constraint():
    torch.manual_seed(0)
    grans = [2, 4, 8]
    raw = torch.randn(500, 8) * torch.linspace(0.1, 9.0, 8) + 3.0
    weights = torch.rand(500) ** 3  # a skewed source distribution
    code = nested_power_normalize(raw, weights, grans, peak=1e9)
    p = (weights / weights.sum()).unsqueeze(1)
    assert torch.allclose((p * code).sum(0), torch.zeros(8), atol=1e-4)  # no power wasted on a mean
    for sl in shell_slices(grans):
        assert float((p * code[:, sl] ** 2).sum()) / (sl.stop - sl.start) == pytest.approx(1.0, abs=1e-4)
    for m in grans:  # every prefix meets the same average-power constraint
        assert float((p * code[:, :m] ** 2).sum()) / m == pytest.approx(1.0, abs=1e-4)
    limited = nested_power_normalize(raw, weights, grans, peak=2.0)
    for sl in shell_slices(grans):
        width = sl.stop - sl.start
        assert float((limited[:, sl] ** 2).sum(1).max()) <= 2.0 * width * (1 + 1e-5)  # per-codeword peak limit
        assert float((p * limited[:, sl] ** 2).sum()) / width <= 1.0 + 1e-5  # clamping never raises the average


def test_awgn_noise_power_matches_snr_and_is_reproducible():
    x = torch.zeros(4000, 16)
    for snr in (0.0, 10.0, 20.0):
        y, eff = Channel("awgn", seed=3)(x, snr)
        assert abs(float(y.var()) - 10 ** (-snr / 10)) / 10 ** (-snr / 10) < 0.05
        assert torch.allclose(eff, torch.full((4000,), snr))
    a, _ = Channel("rayleigh", seed=5)(x[:8], 10.0)
    b, _ = Channel("rayleigh", seed=5)(x[:8], 10.0)
    assert torch.equal(a, b)


def test_rayleigh_gain_statistics():
    eff = Channel("rayleigh", seed=1).effective_snr_db(200000, 10.0)
    gain2 = 10 ** ((eff - 10.0) / 10)
    assert abs(float(gain2.mean()) - 1.0) < 0.02  # E|h|^2 = 1
    assert float(gain2.min()) >= Channel.MIN_GAIN**2 - 1e-9


def test_capacity():
    assert capacity_real(0.0) == pytest.approx(0.5)
    assert capacity_real(10.0) == pytest.approx(0.5 * math.log2(11.0))
    assert float(capacity_real(torch.tensor([0.0]))[0]) == pytest.approx(0.5)


# ------------------------------------------------------------ quantisation
def test_quantisation_cost_and_error_bound():
    torch.manual_seed(0)
    delta = torch.randn(10, 30)
    for bits in (4, 8):
        q, cost = quantize_delta(delta, bits)
        assert cost == 32 + delta.numel() * bits
        step = float(delta.abs().max()) / (2 ** (bits - 1) - 1)
        assert float((q - delta).abs().max()) <= step / 2 + 1e-6
    q, cost = quantize_delta(delta, 8, topk=0.1)
    kept = int((q != 0).sum())
    assert kept <= 30 and cost == 32 + 30 * (8 + math.ceil(math.log2(300)))
    q, cost = quantize_delta(torch.zeros(7), 8)
    assert cost == 1 and float(q.abs().sum()) == 0.0
    q, cost = quantize_delta(delta, 0)
    assert torch.equal(q, delta) and cost == 32 * delta.numel()
    q, cost = quantize_delta(delta, 1)
    assert set(torch.unique(q.abs()).tolist()) == {float(delta.abs().max())}


# -------------------------------------------------------- nested optimiser
def _levels():
    fast = torch.nn.Parameter(torch.zeros(3))
    med = torch.nn.Parameter(torch.zeros(4))
    slow = torch.nn.Parameter(torch.zeros(5))
    levels = [
        Level("fast", "tx", [("fast", fast)], period=1, lr=0.1),
        Level("med", "rx", [("med", med)], period=2, lr=0.1),
        Level("slow", "rx", [("slow", slow)], period=4, lr=0.1, ema=True),
    ]
    return levels, (fast, med, slow)


def test_levels_update_only_at_their_period_and_ledger_is_exact():
    levels, (fast, med, slow) = _levels()
    opt = NestedOptimizer(levels, sync_bits=8, rule="chunk")
    changes = {"fast": [], "med": [], "slow": []}
    for t in range(1, 9):
        before = [p.detach().clone() for p in (fast, med, slow)]
        loss = ((fast - 1) ** 2).sum() + ((med - 1) ** 2).sum() + ((slow - 1) ** 2).sum()
        loss.backward()
        opt.step(t)
        for name, old, new in zip(changes, before, (fast, med, slow)):
            if not torch.equal(old, new):
                changes[name].append(t)
        assert all(p.grad is None for p in (fast, med, slow))
    assert changes == {"fast": list(range(1, 9)), "med": [2, 4, 6, 8], "slow": [4, 8]}
    assert opt.ledger.events == {"med": 4, "slow": 2}  # the tx level is never charged
    assert opt.ledger.bits["med"] == 4 * (32 + 4 * 8) and opt.ledger.bits["slow"] == 2 * (32 + 5 * 8)
    assert opt.expected_bits_per_update(levels[0]) == 0 and opt.expected_bits_per_update(levels[1]) == 32 + 4 * 8
    cumulative = opt.ledger.cumulative(8)
    assert cumulative[0] == 0 and cumulative[-1] == opt.ledger.total_bits and cumulative == sorted(cumulative)


def test_local_rule_learns_every_step_and_refreshes_the_receiver_at_the_period():
    levels, (fast, med, slow) = _levels()
    changed = []
    opt = NestedOptimizer(levels, sync_bits=8, rule="local", on_change=lambda: changed.append(1))
    receiver_changes = {"med": [], "slow": []}
    for t in range(1, 9):
        with opt.receiver_view():
            before = {"med": med.detach().clone(), "slow": slow.detach().clone()}
        working = med.detach().clone()
        loss = ((fast - 1) ** 2).sum() + ((med - 1) ** 2).sum() + ((slow - 1) ** 2).sum()
        loss.backward()
        opt.step(t)
        assert not torch.equal(med.detach(), working)  # the transmitter's copy learns from every message
        with opt.receiver_view():
            for name, param in (("med", med), ("slow", slow)):
                if not torch.equal(before[name], param.detach()):
                    receiver_changes[name].append(t)
        # the two copies coincide exactly when the level has just been shipped
        assert opt.is_dirty("med") == (t % 2 != 0) and opt.is_dirty("slow") == (t % 4 != 0)
    assert receiver_changes == {"med": [2, 4, 6, 8], "slow": [4, 8]}
    assert opt.in_sync and changed
    assert opt.ledger.events == {"med": 4, "slow": 2}
    assert opt.ledger.bits["med"] == 4 * (32 + 4 * 8) and opt.ledger.bits["slow"] == 2 * (32 + 5 * 8)
    # same budget as the chunked rule, but the receiver has moved further in the same time
    chunk_levels, (_, chunk_med, _) = _levels()
    chunk = NestedOptimizer(chunk_levels, sync_bits=8, rule="chunk")
    for t in range(1, 9):
        (((chunk_levels[0].params[0][1] - 1) ** 2).sum() + ((chunk_med - 1) ** 2).sum() + ((chunk_levels[2].params[0][1] - 1) ** 2).sum()).backward()
        chunk.step(t)
    assert chunk.ledger.total_bits == opt.ledger.total_bits
    assert (med.detach() - 1).abs().mean().item() < (chunk_med.detach() - 1).abs().mean().item()


def test_receiver_view_restores_the_working_copy_and_rejects_unknown_rules():
    p = torch.nn.Parameter(torch.zeros(3))
    opt = NestedOptimizer([Level("l", "rx", [("p", p)], period=5, lr=0.1)], rule="local")
    (p - 1).pow(2).sum().backward()
    opt.step(1)
    working = p.detach().clone()
    with opt.receiver_view():
        assert torch.equal(p.detach(), torch.zeros(3))  # nothing has been shipped yet
    assert torch.equal(p.detach(), working) and opt.ledger.total_bits == 0
    with pytest.raises(ValueError):
        NestedOptimizer([], rule="other")


def test_chunked_update_uses_the_mean_of_accumulated_gradients():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = NestedOptimizer([Level("l", "tx", [("p", p)], period=3, lr=0.5)], grad_clip=0.0, rule="chunk")
    seen = []
    orig = opt._opt["l"].step

    def spy():
        seen.append(float(p.grad))
        return orig()

    opt._opt["l"].step = spy
    for t, g in enumerate([1.0, 2.0, 6.0], start=1):
        (p * g).sum().backward()
        opt.step(t)
    assert seen == [pytest.approx(3.0)]


def test_receiver_update_is_exactly_the_quantised_delta():
    p = torch.nn.Parameter(torch.zeros(50))
    opt = NestedOptimizer([Level("rx", "rx", [("p", p)], period=1, lr=0.1)], sync_bits=3)
    (p * torch.linspace(-1, 1, 50)).sum().backward()
    opt.step(1)
    levels = 2 ** (3 - 1) - 1
    scaled = p.detach() / p.detach().abs().max() * levels
    assert torch.allclose(scaled, scaled.round(), atol=1e-5)  # every value sits on the quantiser grid


# ----------------------------------------------------------------- metrics
def test_bleu_and_exact_match():
    ref = "the council adopts the budget today"
    assert sentence_bleu(ref, ref, 4) == pytest.approx(1.0)
    assert sentence_bleu(ref, "completely unrelated words here now", 1) == 0.0
    assert sentence_bleu(ref, "", 1) == 0.0
    # unigram precision 3/4 with brevity penalty exp(1 - 6/4)
    assert sentence_bleu(ref, "the council adopts nothing", 1) == pytest.approx(0.75 * math.exp(1 - 6 / 4))
    scores = bleu_scores([ref, ref], [ref, "the council rejects the budget today"])
    assert scores["bleu1"] > scores["bleu2"] > scores["bleu3"] > scores["bleu4"] == pytest.approx(0.5)
    assert exact_match(["A b."], ["a B ."]) == 1.0 and exact_match(["a b"], ["a c"]) == 0.0


def test_ece():
    rng = np.random.default_rng(0)
    conf = rng.uniform(size=200000)
    correct = rng.uniform(size=200000) < conf
    assert expected_calibration_error(conf, correct) < 0.01
    assert expected_calibration_error(np.full(1000, 0.9), np.zeros(1000)) == pytest.approx(0.9)


# ------------------------------------------------------------------- stats
def test_statistics():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert paired_permutation(a, a) == pytest.approx(1.0)
    assert paired_permutation(a + 1.0, a) == pytest.approx(2 / 2**5)  # exact enumeration, 5 pairs
    boot = paired_bootstrap(a + 0.5, a)
    assert boot["delta"] == pytest.approx(0.5) and boot["ci_low"] == pytest.approx(0.5)
    assert equivalence(a + 0.001, a, margin=0.01)["equivalent"]
    assert not equivalence(a + 0.5, a, margin=0.01)["equivalent"]
    assert mean_sd([1.0, 3.0]) == (2.0, pytest.approx(math.sqrt(2.0)))


# -------------------------------------------------------------- link budget
def test_link_budget():
    assert sync_symbols(100.0, 0.0) == pytest.approx(200.0)  # capacity 0.5 bit / real symbol at 0 dB
    assert net_semantic_rate(64, 0.0, 10, 100, 10.0) == 64
    assert net_semantic_rate(64, 1000.0, 10, 100, 0.0) == pytest.approx(64 + 2000 / 1000)
    assert overhead_fraction(64, 0.0, 10, 100, 10.0) == 0.0
    # adapting pays once traffic exceeds sync symbols per step / payload saving per message
    n_star = crossover_traffic(1000.0, 32, 0.0)
    assert n_star == pytest.approx(2000 / 32)
    assert net_semantic_rate(64, 1000.0, 1, n_star, 0.0) == pytest.approx(64 + 32)
    assert crossover_traffic(1000.0, 0, 0.0) == math.inf


def test_separation_reference():
    assert deflate_bits("a") < deflate_bits("the base station allocates resource blocks " * 3)
    bits = np.array([10.0, 100.0])
    assert delivered(bits, 20, np.array([0.0, 0.0])).tolist() == [True, False]  # 20 symbols * 0.5 bit
    assert delivered(bits, 200, np.array([0.0, 0.0])).tolist() == [True, True]
    assert delivered(bits, np.array([20, 400]), np.array([0.0, 0.0])).tolist() == [True, True]  # per-message budgets
    # finite block length: strictly below capacity, converging to it per symbol as the block grows
    for snr in (0.0, 10.0):
        gaps = [1.0 - float(max_bits(m, snr, eps=1e-3)) / float(max_bits(m, snr)) for m in (16, 256, 65536)]
        assert gaps[0] > gaps[1] > gaps[2] > 0.0 and gaps[2] < 0.03
    assert float(max_bits(100, 10.0, eps=1e-3)) < float(max_bits(100, 10.0, eps=1e-1))
    assert float(max_bits(16, 0.0, eps=1e-3)) == 0.0  # clipped, never negative
    assert float(max_bits(2, -30.0, eps=1e-6)) <= float(max_bits(2, -30.0))  # never above capacity
    # dispersion check at 0 dB: V = 3/8 * log2(e)^2
    from scipy.stats import norm
    expected = 64 * 0.5 - np.sqrt(64 * 0.375 * np.log2(np.e) ** 2) * norm.isf(1e-3) + 0.5 * np.log2(64)
    assert float(max_bits(64, 0.0, eps=1e-3)) == pytest.approx(expected)


# ----------------------------------------------------------------- streams
def test_hier_stream_structure(cfg):
    corpora = {n: get_corpus(n, cfg["data"]) for n in cfg["stream"]["domains"]}
    items = list(HierStream(corpora, cfg["stream"], seed=0))
    assert len(items) == 2 * 6 and [i.t for i in items] == list(range(1, 13))
    assert [i.domain for i in items] == ["toy_b"] * 6 + ["toy_c"] * 6
    assert [i.t for i in items if i.regime_end] == [6, 12]
    assert all(len(i.texts) == 4 and len(i.starts) == 4 and 0.0 <= i.snr_db <= 20.0 for i in items)
    assert items[0].topic != items[3].topic  # medium band refreshed after tau_med steps
    again = list(HierStream(corpora, cfg["stream"], seed=0))
    assert [i.texts for i in again] == [i.texts for i in items]
    other = list(HierStream(corpora, cfg["stream"], seed=1))
    assert [i.texts for i in other] != [i.texts for i in items]
    train = set(corpora["toy_b"].texts("train")) | set(corpora["toy_c"].texts("train"))
    assert all(text in train for item in items for text in item.texts)
    # every conversation walks through whole documents, in order
    assert all(items[0].starts) and all(items[6].starts)  # the stream, and every new domain, open new documents
    assert not all(all(i.starts) for i in items)  # some messages continue a document
    docs = {}
    for name in corpora:
        for doc in corpora[name].documents("train"):
            for position, record in enumerate(doc):
                docs[record["text"]] = (name, record["doc"], position)
    for talk in range(4):
        for before, after in zip(items, items[1:]):
            if not after.starts[talk]:  # a continued message is the next sentence of the same document
                a, b = docs[before.texts[talk]], docs[after.texts[talk]]
                assert a[:2] == b[:2] and b[2] == a[2] + 1
            else:
                assert docs[after.texts[talk]][2] == 0
    probes = build_probes(corpora, cfg["stream"])
    assert set(probes) == {"toy_b", "toy_c"} and all(len(v) == 6 for v in probes.values())
    assert all(p.text not in train for items_ in probes.values() for p in items_)  # probes are held out
    assert any(p.history for items_ in probes.values() for p in items_)
    for p in (p for items_ in probes.values() for p in items_ if p.history):  # the history is the start of the same document
        name, doc, position = None, None, len(p.history)
        test_docs = [d for n in corpora for d in corpora[n].documents("test")]
        assert any([r["text"] for r in d[: position + 1]] == p.history + [p.text] for d in test_docs)


def test_temporal_stream_is_chronological(cfg):
    scfg = dict(cfg["stream"], kind="temporal")
    corpus = get_corpus("toy_b", cfg["data"])
    stream = TemporalStream(corpus, scfg, seed=0)
    year = {r["text"]: r["time"] for r in corpus.records}
    items = list(stream)
    means = [np.mean([int(year[t]) for t in item.texts]) for item in items]
    assert len(items) == 8 and means[0] < means[-1]
    assert [i.domain for i in items] == ["era0"] * 4 + ["era1"] * 4
    assert [i.t for i in items if i.regime_end] == [4, 8]
    assert all(items[0].starts) and all(len(i.starts) == len(i.texts) for i in items)
    probes = build_probes({"toy_b": corpus}, scfg)
    assert set(probes) == {"era0", "era1"} and all(hasattr(p, "history") for p in probes["era0"])


def test_documents_group_consecutive_sentences(cfg):
    corpus = get_corpus("toy_a", cfg["data"])
    docs = corpus.documents("train")
    assert sum(len(d) for d in docs) == len(corpus.split("train"))
    assert all(len({r["doc"] for r in d}) == 1 for d in docs) and max(len(d) for d in docs) > 1
    assert len({d[0]["doc"] for d in docs}) == len(docs)


def test_capping_a_corpus_keeps_whole_documents():
    raw = [{"text": f"Sentence number {i} of document {d} is here.", "topic": "t", "time": "", "doc": d} for d in range(50) for i in range(4)]
    kept = process("toy_cap", raw, 4, 30, max_sentences=40)
    sizes = {}
    for r in kept:
        sizes[r["doc"]] = sizes.get(r["doc"], 0) + 1
    assert 40 <= len(kept) < 44 and set(sizes.values()) == {4}  # no document is cut


def test_snr_trace_is_bounded_and_correlated():
    trace = snr_trace(5000, {"mean": 10.0, "rho": 0.95, "sigma": 4.0, "min": 0.0, "max": 20.0}, np.random.default_rng(0))
    assert trace.min() >= 0.0 and trace.max() <= 20.0 and abs(trace.mean() - 10.0) < 1.0
    assert np.corrcoef(trace[:-1], trace[1:])[0, 1] > 0.8


# ------------------------------------------------------------------- misc
def test_schedules():
    assert lr_at(0, 100, 10) == pytest.approx(0.1) and lr_at(9, 100, 10) == pytest.approx(1.0)
    assert lr_at(99, 100, 10) < 0.01
    rng = np.random.default_rng(0)
    picked = sample_grans([8, 16, 32, 64], 2, rng, always=16)
    assert picked[0] == 16 and len(set(picked)) == 2
    assert sample_grans([8, 16], 5, rng) and len(sample_grans([8, 16], 5, rng)) == 2


def test_calibrator_and_stopping_rule():
    rng = np.random.default_rng(0)
    # an over-confident decoder: true accuracy is sigmoid(0.5 * logit(conf))
    conf = rng.uniform(0.02, 0.98, size=20000)
    logit = np.log(conf / (1 - conf))
    labels = (rng.uniform(size=20000) < 1 / (1 + np.exp(-0.5 * logit))).astype(float)
    feats = features(conf, np.zeros_like(conf), np.zeros_like(conf), 4)
    assert feats.shape == (20000, 2)
    prob = Calibrator().fit(feats, labels).predict(feats)
    assert expected_calibration_error(conf, labels) > 0.08
    assert expected_calibration_error(prob, labels) < 0.02
    # a decoder that is already calibrated is left alone
    labels = (rng.uniform(size=20000) < conf).astype(float)
    prob = Calibrator().fit(feats, labels).predict(feats)
    assert np.abs(prob - conf).mean() < 0.02

    quality = np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    prob = np.array([[0.2, 0.9, 1.0], [0.1, 0.3, 0.95], [0.9, 0.9, 0.9]])  # an informative confidence
    curve = stopping_curve(prob, quality, [8, 16, 32], np.array([0.0, 0.8, 1.1]))
    assert curve[0]["symbols"] == 8 and curve[-1]["symbols"] == 32
    assert curve[1]["symbols"] == pytest.approx((16 + 32 + 8) / 3) and curve[1]["quality"] == 1.0
    refs = reference_policies(quality, [8, 16, 32])
    assert refs["oracle"]["symbols"] == pytest.approx((16 + 32 + 8) / 3) and refs["oracle"]["quality"] == 1.0
    assert [f["m"] for f in refs["fixed"]] == [8, 16, 32] and [f["quality"] for f in refs["fixed"]] == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert interpolate_fixed(refs["fixed"], 12.0) == pytest.approx(0.5)  # time-sharing between m=8 and m=16
    control = random_matched(np.array([1, 2, 0]), quality, [8, 16, 32], seed=0)
    assert control["symbols"] == pytest.approx((16 + 32 + 8) / 3)


def test_symbols_needed_for_a_target_accuracy():
    symbols, accuracy = [2.0, 4.0, 8.0], [0.5, 0.7, 0.9]
    assert symbols_at_accuracy(symbols, accuracy, 0.8) == pytest.approx(6.0)
    assert symbols_at_accuracy(symbols, accuracy, 0.7) == pytest.approx(4.0)
    assert symbols_at_accuracy(symbols, accuracy, 0.3) == 2.0  # already exceeded at the cheapest point
    assert symbols_at_accuracy(symbols, accuracy, 0.95) == math.inf  # never reached
    assert symbols_at_accuracy([8.0, 2.0, 4.0], [0.9, 0.5, 0.7], 0.8) == pytest.approx(6.0)  # order does not matter
    assert symbols_at_accuracy([], [], 0.5) == math.inf


def test_stop_levels_pick_the_first_confident_shell():
    conf = torch.tensor([[0.2, 0.6, 0.95], [0.91, 0.5, 0.99], [0.1, 0.2, 0.3]])
    assert stop_levels(conf, 0.9).tolist() == [2, 0, 2]  # the last shell when never confident
    assert stop_levels(conf, 0.5).tolist() == [1, 0, 2]
    assert stop_levels(conf, 0.0).tolist() == [0, 0, 0]


def test_block_bootstrap_of_paired_differences():
    rng = np.random.default_rng(0)
    shifted = [0.2 + 0.5 * rng.standard_normal(200) for _ in range(3)]
    ci = block_bootstrap(shifted, block=25)
    assert ci["n_blocks"] == 24 and ci["ci_low"] < ci["delta"] < ci["ci_high"] and ci["excludes_zero"] and ci["delta"] == pytest.approx(0.2, abs=0.1)
    null = [0.5 * rng.standard_normal(200) for _ in range(3)]
    assert not block_bootstrap(null, block=25)["excludes_zero"]
    assert block_bootstrap([np.ones(60)], block=25)["n_blocks"] == 2  # an incomplete last block is dropped
    with pytest.raises(ValueError):
        block_bootstrap([np.ones(10)], block=25)


# ------------------------------------------------- refinement signalling, statistics, text metrics (v0.3)
def test_feedback_meter_counts_one_decision_per_open_token_and_round():
    from dnsc.refine import FeedbackMeter, binary_entropy, feedback_bits_per_token

    meter = FeedbackMeter(4)  # shells 0..3: at most three decisions per token
    stop = torch.tensor([0, 1, 3, 3])
    meter.add_stops(stop)
    assert meter.open.tolist() == [4.0, 3.0, 2.0] and meter.cont.tolist() == [3.0, 2.0, 2.0]
    assert meter.bitmap_bits() == pytest.approx((1 + 2 + 3 + 3) / 4)
    assert feedback_bits_per_token(stop, 4).tolist() == [1.0, 2.0, 3.0, 3.0]
    expected = (4 * binary_entropy(3 / 4) + 3 * binary_entropy(2 / 3) + 2 * binary_entropy(1.0)) / 4
    assert meter.entropy_bits() == pytest.approx(expected) and meter.entropy_bits() <= meter.bitmap_bits()
    assert binary_entropy(0.5) == pytest.approx(1.0) and binary_entropy(0.0) == 0.0 and binary_entropy(1.0) == 0.0

    late = FeedbackMeter(4)  # the first round already sends shell 1: one decision fewer
    late.add_stops(torch.tensor([1, 3, 3]), start=1)
    assert late.open.tolist() == [0.0, 3.0, 2.0] and late.bitmap_bits() == pytest.approx((1 + 2 + 2) / 3)
    assert feedback_bits_per_token(torch.tensor([1, 3, 3]), 4, start=1).tolist() == [1.0, 2.0, 2.0]

    both = FeedbackMeter(4)
    both.merge(meter)
    both.merge(late)
    assert both.tokens == 7 and both.open.tolist() == [4.0, 6.0, 4.0]
    single = FeedbackMeter(1)  # one shell only: nothing to ask for
    single.add_stops(torch.tensor([0, 0]))
    assert single.bitmap_bits() == 0.0 and single.entropy_bits() == 0.0


def test_stop_levels_respects_the_first_shell_and_rounds_are_per_message():
    from dnsc.refine import rounds_per_message, stop_levels

    conf = torch.tensor([[0.95, 0.2, 0.99, 0.99], [0.1, 0.2, 0.3, 0.4], [0.1, 0.92, 0.5, 0.99]])
    assert stop_levels(conf, 0.9).tolist() == [0, 3, 1]
    assert stop_levels(conf, 0.9, start=1).tolist() == [2, 3, 1]  # shell 0 is not an option any more
    assert stop_levels(conf, 0.9, start=3).tolist() == [3, 3, 3]
    assert conf.tolist()[0][0] == pytest.approx(0.95)  # the input is not modified
    stop = torch.tensor([0, 2, 1, 1])
    assert rounds_per_message(stop, [2, 2]) == pytest.approx((3 + 2) / 2)  # a message needs as many rounds as its slowest token
    assert rounds_per_message(stop, [2, 2], start=1) == pytest.approx((2 + 1) / 2)


def test_seed_level_tests_state_what_they_can_show():
    from dnsc.stats import block_equivalence, min_attainable_p, paired_permutation

    assert min_attainable_p(3) == pytest.approx(0.25) and min_attainable_p(6) == pytest.approx(0.03125) and min_attainable_p(1) == 1.0
    # three seeds, all in favour of one method: the exact test still cannot go below its floor
    assert paired_permutation([1.0, 1.1, 1.2], [0.0, 0.1, 0.2]) == pytest.approx(0.25)
    assert paired_permutation([1.0, 1.1, 1.2, 1.3, 1.4, 1.5], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]) == pytest.approx(0.03125)
    rng = np.random.default_rng(0)
    tight = [rng.normal(0.0, 0.01, size=200) for _ in range(3)]
    shifted = [rng.normal(0.2, 0.01, size=200) for _ in range(3)]
    assert block_equivalence(tight, margin=0.05)["equivalent"] and not block_equivalence(shifted, margin=0.05)["equivalent"]
    wide = block_equivalence([rng.normal(0.0, 2.0, size=200)], margin=0.05)
    assert not wide["equivalent"] and wide["ci_low"] < 0 < wide["ci_high"]  # too noisy to call: not "equivalent"


def test_chrf_and_word_error_rate():
    from dnsc.metrics import chrf, fidelity, word_error_rate

    refs = ["the council adopts the budget", "a late goal in extra time"]
    assert chrf(refs, refs) == pytest.approx(1.0) and word_error_rate(refs, refs) == 0.0
    one_wrong = ["the council adopts the budgets", "a late goal in extra time"]
    assert 0.5 < chrf(refs, one_wrong) < 1.0  # partial credit inside a word, unlike BLEU-4
    assert word_error_rate(refs, one_wrong) == pytest.approx(1 / 11)
    assert word_error_rate(["a b c"], ["a c"]) == pytest.approx(1 / 3) and word_error_rate(["a b c"], [""]) == pytest.approx(1.0)
    out = fidelity(refs, one_wrong)
    assert {"bleu4", "sacrebleu", "exact", "chrf", "wer"} <= set(out) and out["exact"] == 0.5


def test_code_revision_is_read_from_the_release_file_outside_git(tmp_path, monkeypatch):
    import dnsc.utils as utils

    package = tmp_path / "dnsc"
    package.mkdir()
    (tmp_path / "REVISION").write_text("abc1234 built 2026-10-13\n", encoding="utf-8")
    monkeypatch.setattr(utils, "__file__", str(package / "utils.py"))
    assert utils.git_revision() == "abc1234"
    (tmp_path / "REVISION").unlink()
    assert utils.git_revision() == "unknown"
