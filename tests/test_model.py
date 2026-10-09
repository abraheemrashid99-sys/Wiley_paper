"""Codec, LoRA and end-to-end integration tests on the tiny offline backbone."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from dnsc.channel import Channel
from dnsc.classical import lm_bits
from dnsc.codec import GROUPS, load_codec
from dnsc.lora import LoRALinear, inject_lora
from dnsc.online import METHODS, build_levels, matched_period, run_stream
from dnsc.refine import run_refinement
from dnsc.train import evaluate_gen, evaluate_tf, pretrain

TEXTS = ["The council adopts the budget on trade today.", "The striker scores a late goal in extra time.", "The scheduler allocates resource blocks in every slot."]
ROOT = Path(__file__).resolve().parents[1]


def test_lora_is_identity_at_init_and_sides_are_independent():
    torch.manual_seed(0)
    base = torch.nn.Linear(6, 5)
    layer = LoRALinear(base, r=2, alpha=4, dropout=0.0, sides=("tx", "rx"))
    x = torch.randn(3, 6)
    layer.active = "tx"
    assert torch.equal(layer(x), base(x))  # B is zero-initialised
    with torch.no_grad():
        layer.lora_B["tx"].fill_(0.3)
    assert not torch.allclose(layer(x), base(x))
    layer.active = "rx"
    assert torch.equal(layer(x), base(x))  # the other side is untouched
    layer.active = None
    assert torch.equal(layer(x), base(x))


def test_lora_last_n_restricts_layers():
    from transformers import LlamaConfig, LlamaForCausalLM

    lm = LlamaForCausalLM(LlamaConfig(vocab_size=50, hidden_size=16, intermediate_size=32, num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2))
    wrapped = inject_lora(lm, ["q_proj", "v_proj"], r=2, alpha=4, dropout=0.0, tx_last_n=0, rx_last_n=1)
    assert len(wrapped) == 6
    assert sum("rx" in w.lora_A for w in wrapped) == 2 and sum("tx" in w.lora_A for w in wrapped) == 6


def test_codec_shapes_power_and_groups(codec):
    batch = codec.tokenize(TEXTS)
    assert batch.ids.shape[0] == 3 and int(batch.lens.max()) == batch.ids.shape[1]
    assert all(int(batch.ids[i, batch.lens[i] - 1]) == codec.tok.eos_token_id for i in range(3))
    z = codec.encode(batch)
    assert z.shape == (3, codec.L)
    for m in codec.grans:
        assert torch.allclose((z[:, :m] ** 2).sum(1), torch.full((3,), float(m)), atol=1e-3)
    y, eff = codec.transmit(z, 8, 10.0, Channel("awgn", 0))
    assert float(y[:, 8:].detach().abs().sum()) == 0.0 and eff.shape == (3,)
    ce, acc = codec.decode_tf(codec.prompts(y, 8), batch)
    assert ce.shape == (3,) and torch.isfinite(ce).all() and ((acc >= 0) & (acc <= 1)).all()

    groups = codec.groups()
    assert set(groups) == set(GROUPS) and all(groups[g] for g in GROUPS)
    names = [n for g in groups.values() for n, _ in g]
    assert len(names) == len(set(names))
    expected = {n for n, _ in codec.named_parameters() if not n.startswith("lm.") or "lora_" in n}
    assert set(names) == expected  # every non-backbone parameter is in exactly one group


def test_padding_does_not_change_a_sentence_encoding(codec):
    alone = codec.encode(codec.tokenize([TEXTS[1]]))
    padded = codec.encode(codec.tokenize([TEXTS[1], TEXTS[0] + " " + TEXTS[2]]))
    assert torch.allclose(alone[0], padded[0], atol=1e-4)
    batch1, batch2 = codec.tokenize([TEXTS[1]]), codec.tokenize([TEXTS[1], TEXTS[0] + " " + TEXTS[2]])
    y = torch.randn(1, codec.L)
    ce1, _ = codec.decode_tf(codec.prompts(y, 16), batch1)
    ce2, _ = codec.decode_tf(codec.prompts(y.repeat(2, 1), 16), batch2)
    assert torch.allclose(ce1[0], ce2[0], atol=1e-4)


def test_gradients_reach_both_ends_through_the_channel(codec):
    for p in codec.parameters():
        p.requires_grad_(True)
    batch = codec.tokenize(TEXTS)
    ce, _ = codec.loss_at(codec.encode(batch), batch, 16, 10.0, Channel("awgn", 0))
    ce.mean().backward()
    groups = codec.groups()
    for name in ("tx_fast", "tx_lora", "rx_adapter", "rx_static"):
        assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for _, p in groups[name]), name
    # rx LoRA B matrices receive gradient (A is shielded by the zero-initialised B)
    assert any("lora_B" in n and p.grad is not None and float(p.grad.abs().sum()) > 0 for n, p in groups["rx_lora"])


def test_generation_matches_teacher_forcing_argmax(codec):
    """The cached greedy decoder must agree with a full forward pass."""
    batch = codec.tokenize(TEXTS[:1])
    y, _ = codec.transmit(codec.encode(batch), 16, 20.0, Channel("awgn", 0))
    prompts = codec.prompts(y, 16)
    decoded = codec.generate(prompts, max_new=6)
    ids = decoded.token_ids[0]
    assert 0.0 < float(decoded.min_prob[0]) <= 1.0 and float(decoded.mean_entropy[0]) >= 0.0
    if ids:  # re-score the generated prefix without the cache
        codec._side("rx")
        x = torch.cat([prompts, codec._embed(torch.tensor([ids]))], dim=1)
        hidden = codec.lm.base_model(inputs_embeds=x).last_hidden_state
        greedy = codec._head(hidden[:, codec.P - 1 : -1]).argmax(-1)[0].tolist()
        assert greedy == ids


def test_codec_can_memorise_a_small_set(cfg):
    """Functional check of the whole differentiable path: sentence -> symbols
    -> noisy channel -> soft prompts -> sentence. If information did not flow
    through the channel bottleneck this could not reach high accuracy."""
    from dnsc.codec import build_codec
    from dnsc.data import get_corpus

    torch.manual_seed(0)
    cfg["backbone"]["tiny"] = {"hidden": 64, "layers": 2, "heads": 4, "vocab": 400}
    cfg["codec"]["max_tokens"] = 48  # long enough that no sentence is truncated
    model = build_codec(cfg, torch.device("cpu"))
    texts = get_corpus("toy_a", cfg["data"]).texts("train")[:12]
    batch = model.tokenize(texts)
    channel = Channel("awgn", 0)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-3)
    for _ in range(300):
        ce, acc = model.loss_at(model.encode(batch), batch, 16, 20.0, channel)
        optimizer.zero_grad()
        ce.mean().backward()
        optimizer.step()
    model.eval()
    assert evaluate_tf(model, texts, 16, 20.0, Channel("awgn", 1))["tok_acc"] > 0.97
    assert evaluate_gen(model, texts, 16, 20.0, Channel("awgn", 2))["exact"] > 0.8
    # the same decoder fed pure noise cannot know which sentence was sent
    from dnsc.metrics import exact_match

    noise = model.mask_prefix(torch.randn(12, model.L), 16)
    assert exact_match(texts, model.generate(model.prompts(noise, 16), max_new=48).texts) < 0.3


def test_checkpoint_round_trip(cfg, codec, checkpoint):
    restored = load_codec(cfg, checkpoint, torch.device("cpu"))
    restored.eval()
    a = evaluate_tf(codec, TEXTS, 8, 10.0, Channel("awgn", 4))
    b = evaluate_tf(restored, TEXTS, 8, 10.0, Channel("awgn", 4))
    assert a["ce_bits"] == pytest.approx(b["ce_bits"], abs=1e-5)
    with pytest.raises(FileNotFoundError):
        load_codec(cfg, checkpoint.parent / "missing", torch.device("cpu"))


def test_evaluation_and_classical_reference(codec):
    gen = evaluate_gen(codec, TEXTS, 16, 10.0, Channel("awgn", 0), return_texts=True)
    assert len(gen["hyps"]) == 3 and 0.0 <= gen["bleu1"] <= 1.0 and "sim" not in gen
    bits = lm_bits(codec, TEXTS)
    assert bits.shape == (3,) and (bits > 0).all()
    # an untrained LM is near uniform: about log2(vocab) bits per token
    tokens = codec.tokenize(TEXTS).lens.numpy()
    assert np.allclose(bits / tokens, np.log2(len(codec.tok)), rtol=0.25)


def test_level_schedules(cfg, codec):
    ncfg = cfg["nested"]
    assert build_levels(codec, "frozen", ncfg) == []
    nested = build_levels(codec, "nested", ncfg)
    assert [(lv.name, lv.side, lv.period) for lv in nested] == [("tx_fast", "tx", 1), ("tx_lora", "tx", 2), ("rx_adapter", "rx", 2), ("rx_lora", "rx", 3)]
    assert nested[-1].ema and not build_levels(codec, "nested_no_ema", ncfg)[-1].ema
    assert all(lv.side == "tx" for lv in build_levels(codec, "tx_only", ncfg))
    assert {lv.period for lv in build_levels(codec, "full_online", ncfg)} == {1}
    single = build_levels(codec, "single_rate", ncfg)
    period = matched_period(codec, ncfg)
    assert {lv.period for lv in single} == {period} and 2 <= period <= 3

    def rate(levels):  # receiver sync bits per step
        return sum((32 * len(lv.params) + 8 * lv.n_params) / lv.period for lv in levels if lv.side == "rx")

    assert rate(single) == pytest.approx(rate(nested), rel=0.35)  # matched budget up to period rounding
    with pytest.raises(ValueError):
        build_levels(codec, "not_a_method", ncfg)


@pytest.mark.parametrize("method", METHODS)
def test_stream_runs_for_every_method(cfg, checkpoint, tmp_path, method):
    out = tmp_path / f"{method}.json"
    result = run_stream(cfg, method, seed=0, ckpt_dir=checkpoint, out_path=out)
    saved = json.loads(out.read_text())
    assert saved["n_steps"] == 12 and len(saved["log"]["tok_acc"]) == 12
    assert len(saved["probes"]) == 3 and set(saved["probes"][0]["domains"]) == {"toy_b", "toy_c"}
    assert len(saved["ledger"]["cumulative"]) == 12 and saved["ledger"]["cumulative"][-1] == saved["summary"]["sync_bits"]
    if method in ("frozen", "tx_only"):
        assert result["summary"]["sync_bits"] == 0 and result["summary"]["net_rate"] == saved["m_op"]
    else:
        assert result["summary"]["sync_bits"] > 0 and result["summary"]["net_rate"] > saved["m_op"]
    assert np.isfinite(saved["log"]["ce_bits"]).all()


def test_frozen_stream_leaves_the_codec_untouched_and_adaptive_changes_it(cfg, checkpoint, tmp_path):
    a = run_stream(cfg, "frozen", 0, checkpoint, tmp_path / "a.json")
    b = run_stream(cfg, "frozen", 0, checkpoint, tmp_path / "b.json")
    assert a["log"]["ce_bits"] == b["log"]["ce_bits"]  # deterministic given the seed
    assert a["probes"][0]["domains"]["toy_b"]["ce_bits"] == pytest.approx(a["probes"][-1]["domains"]["toy_b"]["ce_bits"], abs=1e-6)
    c = run_stream(cfg, "full_online", 0, checkpoint, tmp_path / "c.json")
    assert c["probes"][0]["domains"]["toy_b"]["ce_bits"] != pytest.approx(c["probes"][-1]["domains"]["toy_b"]["ce_bits"], abs=1e-6)
    assert a["log"]["ce_bits"][0] == pytest.approx(c["log"]["ce_bits"][0], abs=1e-5)  # scored before the first update


def test_temporal_stream_runs(cfg, checkpoint, tmp_path):
    cfg["stream"]["kind"] = "temporal"
    result = run_stream(cfg, "nested", 0, checkpoint, tmp_path / "t.json")
    assert result["n_steps"] == 8 and set(result["probes"][-1]["domains"]) == {"era0", "era1"}


def test_refinement_runs(cfg, codec):
    res = run_refinement(codec, TEXTS * 4, TEXTS * 4, 10.0, "rayleigh", cfg["refine"], batch_size=4)
    assert len(res["curve"]) >= 2 and len(res["fixed"]) == 3
    symbols = [c["symbols"] for c in res["curve"]]
    assert symbols == sorted(symbols)  # a stricter threshold never sends fewer symbols
    assert symbols[0] == 4.0 and symbols[-1] == 16.0  # the sweep spans "first shell" to "everything"
    assert 0.0 <= res["ece_calibrated"] <= 1.0 and res["oracle"]["symbols"] <= 16


def test_pretraining_reduces_the_loss(cfg):
    cfg["pretrain"]["steps"] = 60
    cfg["pretrain"]["eval_every"] = 60
    cfg["pretrain"]["lr"] = 3e-3
    cfg["pretrain"]["lora_lr"] = 3e-3
    directory = pretrain(cfg)
    lines = [json.loads(line) for line in (directory / "pretrain_log.jsonl").read_text().splitlines()]
    losses = [row["loss"] for row in lines if "loss" in row]
    assert losses[-1] < 0.7 * losses[0]
    assert (directory / "codec.pt").exists() and (directory / "pretrain_summary.json").exists()


def test_command_line_pipeline(tmp_path):
    """Every script runs end to end from the shell on the offline config."""
    overrides = [f"out_dir={tmp_path / 'run'}", f"data.cache_dir={tmp_path / 'data'}", "pretrain.steps=20", "pretrain.eval_every=20", "stream.tau_slow=8", "stream.cycles=1", "stream.gen_every=4", "eval.n_test=8", "eval.snrs=[10.0]", "refine={quality: bleu1, gamma: 0.5, n_val: 8, n_test: 8}"]

    def run(script: str, *extra: str) -> str:
        cmd = [sys.executable, str(ROOT / "scripts" / script), "--config", str(ROOT / "configs" / "smoke.yaml"), *extra, *overrides]
        done = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT, timeout=900)
        assert done.returncode == 0, f"{script} failed:\n{done.stdout[-2000:]}\n{done.stderr[-2000:]}"
        return done.stdout

    run("check_env.py")
    run("prepare_data.py")
    run("pretrain.py")
    run("eval_rate.py")
    run("run_stream.py", "--methods", "frozen,nested,single_rate", "--seeds", "0,1")
    assert "skip existing" in run("run_stream.py", "--methods", "frozen", "--seeds", "0")
    run("eval_refine.py")
    run("benchmark_speed.py", "--steps", "2", "--repeats", "3")
    run("aggregate.py")
    run("make_figures.py")
    out = tmp_path / "run"
    for name in ("tables/stream_hier.md", "tables/netrate_hier.csv", "tables/rate_toy_a_bleu4.tex", "tables/refine_toy_a.md", "figures/stream_hier.png", "figures/rate_toy_a.pdf", "figures/refine_toy_a.png", "eval/speed.json"):
        assert (out / name).exists(), name


def test_pretrained_backbone_path(cfg, tmp_path):
    """The code path used for real backbones: a causal LM and tokenizer loaded
    by name through ``Auto*``, stored in bfloat16 like SmolLM2, kept frozen,
    and *not* written into the codec checkpoint."""
    from transformers import LlamaConfig, LlamaForCausalLM

    from dnsc.backbone import _build_tiny_tokenizer
    from dnsc.codec import build_codec

    hub = tmp_path / "fake-hub-model"
    tokenizer = _build_tiny_tokenizer(400)
    tokenizer.save_pretrained(str(hub))
    torch.manual_seed(1)
    config = LlamaConfig(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    LlamaForCausalLM(config).to(torch.bfloat16).save_pretrained(str(hub))

    cfg["backbone"]["name"] = str(hub)
    cfg["backbone"]["train_backbone"] = False
    codec = build_codec(cfg, torch.device("cpu"))
    assert all(p.dtype == torch.float32 for p in codec.parameters())
    assert not any(p.requires_grad for n, p in codec.named_parameters() if n.startswith("lm.") and "lora_" not in n)

    directory = pretrain(cfg)
    state = torch.load(directory / "codec.pt", map_location="cpu")["state"]
    assert state and all(not k.startswith("lm.") or "lora_" in k for k in state)  # backbone is not duplicated on disk
    restored = load_codec(cfg, directory, torch.device("cpu"))
    restored.eval()
    result = run_stream(cfg, "nested", 0, directory, tmp_path / "s.json")
    assert result["summary"]["sync_bits"] > 0 and np.isfinite(result["log"]["ce_bits"]).all()
    again = load_codec(cfg, directory, torch.device("cpu"))
    again.eval()
    a = evaluate_tf(restored, TEXTS, 8, 10.0, Channel("awgn", 4))
    b = evaluate_tf(again, TEXTS, 8, 10.0, Channel("awgn", 4))
    assert a["ce_bits"] == pytest.approx(b["ce_bits"], abs=1e-5)  # the stream run did not alter the checkpoint


def test_pretraining_resumes_bit_for_bit(cfg, tmp_path):
    """An interrupted run that is resumed must end exactly where an
    uninterrupted run ends (same weights, same evaluation history)."""
    import copy

    cfg["pretrain"].update({"steps": 8, "eval_every": 4, "log_every": 2, "save_every": 2})
    straight = copy.deepcopy(cfg)
    straight["out_dir"] = str(tmp_path / "straight")
    broken = copy.deepcopy(cfg)
    broken["out_dir"] = str(tmp_path / "broken")

    reference = pretrain(straight)
    directory = pretrain(broken, stop_after=3)  # interrupted after step 3
    assert (directory / "pretrain_state.pt").exists() and not (directory / "codec.pt").exists()
    pretrain(broken, stop_after=2)  # resumed, interrupted again after step 5
    pretrain(broken)  # resumed to the end
    assert (directory / "codec.pt").exists() and not (directory / "pretrain_state.pt").exists()

    a = torch.load(reference / "codec.pt", map_location="cpu")
    b = torch.load(directory / "codec.pt", map_location="cpu")
    assert a["state"].keys() == b["state"].keys()
    for key in a["state"]:
        assert torch.allclose(a["state"][key].float(), b["state"][key].float(), atol=1e-6), key
    assert [row["step"] for row in b["extra"]["history"]] == [4, 8]
    for row_a, row_b in zip(a["extra"]["history"], b["extra"]["history"]):
        assert row_a["8"]["ce_bits"] == pytest.approx(row_b["8"]["ce_bits"], abs=1e-5)

    before = (directory / "codec.pt").stat().st_mtime_ns
    pretrain(broken)  # a finished checkpoint is left alone
    assert (directory / "codec.pt").stat().st_mtime_ns == before
    pretrain(broken, force=True)
    assert (directory / "codec.pt").stat().st_mtime_ns != before


def test_pipeline_runner_plans_every_stage():
    def run(*args: str) -> str:
        cmd = [sys.executable, str(ROOT / "scripts" / "run_pipeline.py"), "--config", "configs/laptop.yaml", "--dry-run", *args]
        done = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT, timeout=60)
        assert done.returncode == 0, done.stderr
        return done.stdout

    everything = run("--stage", "all", "--seeds", "0,1", "threads=4")
    assert everything.count("=====") == 10 and "pretrain.py --config configs/laptop.yaml threads=4" in everything
    assert "--seeds 0,1 stream.kind=temporal threads=4" in everything
    assert run("--stage", "smoke").count("=====") == 8 and run("--stage", "report").count("=====") == 2


def test_real_configs_are_valid():
    from dnsc.config import load_config

    for name in ("smoke", "laptop", "dev", "main", "scale"):
        loaded = load_config(ROOT / "configs" / f"{name}.yaml")
        assert loaded["stream"]["m_op"] in loaded["codec"]["granularities"], name
