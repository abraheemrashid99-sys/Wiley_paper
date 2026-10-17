"""Every number the manuscript quotes, read from the result files (nothing typed).

    python scripts/paper_numbers.py            # writes paper/numbers.json and paper/NUMBERS.md

Sources: results/gpt2_v0.4_six_seeds (primary endpoint), results/decoder_selection
(decoder chosen on validation text), results/gpt2_v0.3_colab_t4 (memory ablations,
pilot protocol, GPU), results/gpt2 (base-corpus link-level evaluation and codebook).
"""

from __future__ import annotations

import csv
import glob
import json
import re

import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def rows(path: str) -> dict[str, dict]:
    with open(ROOT / path, encoding="utf-8") as handle:
        return {r[next(iter(r))]: r for r in csv.DictReader(handle)}


def nums(cell: str) -> list[float]:
    return [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", cell)]


def main() -> None:
    out: dict[str, object] = {}
    src: dict[str, str] = {}

    def put(key: str, value, source: str) -> None:
        out[key] = value
        src[key] = source

    # ---- primary endpoint, six seeds
    six = "results/gpt2_v0.4_six_seeds"
    real = rows(f"{six}/tables/realdecode_hier.csv")
    for m in ("frozen", "context_only", "nested"):
        r = real[m]
        put(f"real.{m}.seeds", int(r["seeds"]), f"{six}/tables/realdecode_hier.csv")
        put(f"real.{m}.steps", int(r["audited steps"]), f"{six}/tables/realdecode_hier.csv")
        put(f"real.{m}.fixed", float(r["fixed prefix"]), f"{six}/tables/realdecode_hier.csv")
        put(f"real.{m}.sel_entropy", float(r["selective + feedback (entropy)"]), f"{six}/tables/realdecode_hier.csv")
        put(f"real.{m}.sel_bitmap", float(r["selective + feedback (bitmap)"]), f"{six}/tables/realdecode_hier.csv")
        for col, key in (("vs frozen fixed [95% CI]", "vs_ref"), ("vs own fixed (what refinement adds)", "vs_own"), ("bitmap vs frozen fixed", "vs_ref_bitmap"), ("fixed prefix vs frozen fixed", "fixed_vs_ref")):
            if r[col] not in ("—", "-"):
                put(f"real.{m}.{key}", nums(r[col]), f"{six}/tables/realdecode_hier.csv")
        put(f"real.{m}.p", r["p (seed-level)"], f"{six}/tables/realdecode_hier.csv")
        put(f"real.{m}.p_holm", r["p (Holm)"], f"{six}/tables/realdecode_hier.csv")
        put(f"real.{m}.per_seed", nums(r["per-seed saving vs frozen fixed (entropy)"]), f"{six}/tables/realdecode_hier.csv")
    emu = rows(f"{six}/tables/emulation_check_hier.csv")
    for m in ("frozen", "context_only", "nested"):
        put(f"emulation_gap.{m}", emu[m]["gap"], f"{six}/tables/emulation_check_hier.csv")
    cost = rows(f"{six}/tables/refine_cost_hier.csv")
    for m in ("frozen", "context_only", "nested"):
        r = cost[m]
        for col, key in (("fixed prefix", "fixed"), ("selective, forward only", "sel_forward"), ("selective + feedback (entropy)", "sel_entropy"), ("selective + feedback (bitmap)", "sel_bitmap"), ("feedback bits/token (entropy)", "fb_entropy_bits"), ("feedback bits/token (bitmap)", "fb_bitmap_bits"), ("rounds/message", "rounds")):
            put(f"tf.{m}.{key}", float(r[col]), f"{six}/tables/refine_cost_hier.csv")
    stream = rows(f"{six}/tables/stream_hier.csv")
    for m in ("frozen", "context_only", "nested"):
        put(f"tf.{m}.prior_bits", stream[m]["prior bits/token"], f"{six}/tables/stream_hier.csv")
        put(f"tf.{m}.sync_kbit_per_step", float(stream[m]["sync kbit/step"]), f"{six}/tables/stream_hier.csv")
        put(f"tf.{m}.ece", stream[m]["ECE"], f"{six}/tables/stream_hier.csv")
        if stream[m]["prior bits: this − nested [95% CI]"] != "-":
            put(f"tf.{m}.dprior_vs_nested", nums(stream[m]["prior bits: this − nested [95% CI]"]), f"{six}/tables/stream_hier.csv")
            put(f"tf.{m}.dsel_vs_nested", nums(stream[m]["selective sym/token: this − nested [95% CI]"]), f"{six}/tables/stream_hier.csv")
            put(f"tf.{m}.p_prior", stream[m]["p (seed-level)"], f"{six}/tables/stream_hier.csv")
    # teacher-forced per-seed readings and matched-seed savings
    runs: dict[str, dict[int, dict]] = {}
    for path in sorted(glob.glob(str(ROOT / six / "stream" / "hier_*.json"))):
        run = json.load(open(path, encoding="utf-8"))
        runs.setdefault(run["method"], {})[run["seed"]] = run["summary"]
    mean = lambda m, k, s: sum(runs[m][x][k] for x in s) / len(s)  # noqa: E731
    for m in ("frozen", "context_only", "nested"):
        seeds = sorted(runs[m])
        ref = mean("frozen", "symbols_at_target_fixed", seeds)
        put(f"tf.{m}.seeds", seeds, f"{six}/stream/*.json")
        put(f"tf.{m}.ref_fixed_same_seeds", ref, f"{six}/stream/*.json")
        for key, name in (("symbols_at_target_fixed", "fixed"), ("symbols_at_target_selective", "sel_forward"), ("symbols_at_target_selective_fb_entropy", "sel_entropy"), ("symbols_at_target_selective_fb", "sel_bitmap")):
            put(f"tf.{m}.{name}_vs_ref_pct", 100 * (mean(m, key, seeds) / ref - 1), f"{six}/stream/*.json")
        put(f"tf.{m}.refine_adds_pct", 100 * (mean(m, "symbols_at_target_selective_fb_entropy", seeds) / mean(m, "symbols_at_target_fixed", seeds) - 1), f"{six}/stream/*.json")
        put(f"tf.{m}.per_seed_saving_pct", [100 * (runs[m][s]["symbols_at_target_selective_fb_entropy"] / runs["frozen"][s]["symbols_at_target_fixed"] - 1) for s in seeds], f"{six}/stream/*.json")
        put(f"tf.{m}.tokens_per_message", mean(m, "tokens_per_message", seeds), f"{six}/stream/*.json")
    unseen = [3, 4, 5]
    put("tf.nested.unseen_saving_pct", 100 * (mean("nested", "symbols_at_target_selective_fb_entropy", unseen) / mean("frozen", "symbols_at_target_fixed", unseen) - 1), f"{six}/stream/*.json")

    # ---- decoder selection (validation)
    sel = json.load(open(ROOT / "results/decoder_selection/decoder_selection.json", encoding="utf-8"))
    put("decoder.decision", sel["decision"], "results/decoder_selection/decoder_selection.json")
    put("decoder.n_messages", sel["n_messages"], "results/decoder_selection/decoder_selection.json")
    put("decoder.n_tokens", sel["n_tokens"], "results/decoder_selection/decoder_selection.json")
    put("decoder.snr_mean", sel["snr_mean"], "results/decoder_selection/decoder_selection.json")
    for name, c in sel["candidates"].items():
        put(f"decoder.{name}", {k: c[k] for k in ("forward", "total_entropy", "total_bitmap", "fixed")}, "results/decoder_selection/decoder_selection.json")

    # ---- memory ablations, pilot protocol (GPU reproduction, single revision)
    pilot = "results/gpt2_v0.3_colab_t4"
    st = rows(f"{pilot}/tables/stream_hier.csv")
    eq = rows(f"{pilot}/tables/equivalence_hier.csv")
    for m, r in st.items():
        put(f"pilot.{m}", {"seeds": int(r["seeds"]), "prior_bits": r["prior bits/token"], "sync_kbit_per_step": float(r["sync kbit/step"]), "dprior_vs_nested_95": r["prior bits: this − nested [95% CI]"], "equivalent_prior": eq.get(m, {}).get("prior bits/token: equivalent within ±0.05", "-"), "dprior_vs_nested_90": eq.get(m, {}).get("prior bits/token: this − nested [90% CI]", "-")}, f"{pilot}/tables/stream_hier.csv, equivalence_hier.csv")

    # ---- base corpus (published pilot, CPU)
    base = "results/gpt2"
    with open(ROOT / f"{base}/tables/rate_europarl_nltk_gen_tok_acc.csv", encoding="utf-8") as handle:
        acc = [r for r in csv.DictReader(handle)]
    put("base.gen_tok_acc_awgn", {f"{r['prior']}@{r['m']}": {"0": float(r["0 dB"]), "10": float(r["10 dB"]), "20": float(r["20 dB"])} for r in acc if r["channel"] == "awgn"}, f"{base}/tables/rate_europarl_nltk_gen_tok_acc.csv")
    with open(ROOT / f"{base}/tables/rate_europarl_nltk_bleu4.csv", encoding="utf-8") as handle:
        bleu = [r for r in csv.DictReader(handle)]
    put("base.bleu4", {f"{r['channel']}:{r['prior']}@{r['m']}": {"0": float(r["0 dB"]), "10": float(r["10 dB"]), "20": float(r["20 dB"])} for r in bleu}, f"{base}/tables/rate_europarl_nltk_bleu4.csv")
    summ = json.load(open(ROOT / f"{base}/base/pretrain_summary.json", encoding="utf-8"))
    put("codebook.channel_only_acc_10dB", summ["history"][-1], f"{base}/base/pretrain_summary.json")
    put("codebook.train_seconds", summ["seconds"], f"{base}/base/pretrain_summary.json")
    fixed8 = json.load(open(ROOT / f"{base}/fixed8/pretrain_summary.json", encoding="utf-8"))
    put("codebook.fixed8_channel_only_acc_10dB", fixed8["history"][-1]["8"], f"{base}/fixed8/pretrain_summary.json")
    with open(ROOT / f"{base}/tables/rate_europarl_nltk_separation.csv", encoding="utf-8") as handle:
        sep = [r for r in csv.DictReader(handle)]
    put("base.separation_lm_fbl_awgn", {r["m"]: {"0": float(r["0 dB"]), "10": float(r["10 dB"]), "20": float(r["20 dB"])} for r in sep if r["system"] == "sep_lm_fbl" and r["channel"] == "awgn"}, f"{base}/tables/rate_europarl_nltk_separation.csv")

    # ---- measured compute (pilot reproduction on a T4 GPU, greedy decoder)
    with open(ROOT / f"{pilot}/tables/cost.csv", encoding="utf-8") as handle:
        put("compute.t4", {r["quantity"]: r["value"] for r in csv.DictReader(handle)}, f"{pilot}/tables/cost.csv")

    # ---- quantities that need the raw runs (per-step audits): unseen seeds, crossover, run times
    import sys
    import math as _math

    sys.path.insert(0, str(ROOT))
    from dnsc.linkbudget import audit_comparison  # noqa: E402

    raw_dir = ROOT / "runs" / "six" / "stream"
    if not raw_dir.exists():
        import tarfile

        raw_dir = ROOT / "runs" / "paper_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(ROOT / six / "raw_streams.tar.gz") as tar:
            tar.extractall(raw_dir)
    raw: dict[str, dict[int, dict]] = {}
    for path in sorted(raw_dir.glob("hier_*.json")):
        run = json.load(open(path, encoding="utf-8"))
        raw.setdefault(run["method"], {})[run["seed"]] = run
    src_raw = f"{six}/raw_streams.tar.gz (per-step audits)"
    for m in ("frozen", "nested"):
        cmp = audit_comparison({s: raw[m][s] for s in unseen}, {s: raw["frozen"][s] for s in unseen}, 0.9, seed=20261015)
        put(f"real.{m}.unseen_vs_ref_pct", [100 * cmp["vs_reference_entropy"][k] for k in ("estimate", "ci_low", "ci_high")], src_raw)
    common = sorted(set(raw["frozen"]) & set(raw["context_only"]) & set(raw["nested"]))
    cap = raw["nested"][common[0]]["summary"]["feedback_capacity"]
    put("link.capacity_10dB_bits_per_real_symbol", cap, src_raw)

    def payload(m: str, seeds: list[int]) -> float:
        return sum(raw[m][s]["summary"]["symbols_at_target_selective_fb_entropy"] * raw[m][s]["summary"]["tokens_per_message"] for s in seeds) / len(seeds)

    def fixed_payload(m: str, seeds: list[int]) -> float:
        return sum(raw[m][s]["summary"]["symbols_at_target_fixed"] * raw[m][s]["summary"]["tokens_per_message"] for s in seeds) / len(seeds)

    bits = sum(raw["nested"][s]["summary"]["sync_bits_per_step"] for s in common) / len(common)
    put("link.crossover_seeds", common, src_raw)
    put("link.sync_bits_per_step", bits, src_raw)
    for ref, value in (("frozen", payload("frozen", common)), ("context_only", payload("context_only", common)), ("frozen_fixed", fixed_payload("frozen", common))):
        mine = payload("nested", common)
        star = bits / (cap * (value - mine)) if value > mine else _math.inf
        put(f"link.crossover_vs_{ref}", {"N_star": star, "asymptote_pct": 100 * (mine / value - 1), "payload_ref": value, "payload_dnsc": mine}, src_raw)
    prior = {m: sum(raw[m][s]["summary"]["mean_prior_bits"] for s in common) / len(common) for m in ("frozen", "context_only", "nested")}
    put("tf.prior_bits_matched_seeds", prior, src_raw)
    put("tf.context_share_of_prior_gain_matched_pct", 100 * (prior["frozen"] - prior["context_only"]) / (prior["frozen"] - prior["nested"]), src_raw)
    # seed-level test of the teacher-forced headline (per-seed cost with signalling against the frozen
    # fixed prefix of the same seed), Holm-corrected over the three refinement receivers
    from dnsc.stats import block_bootstrap, block_equivalence, holm, paired_permutation  # noqa: E402

    tf_p = {}
    for m in ("frozen", "context_only", "nested"):
        seeds = sorted(raw[m])
        tf_p[m] = paired_permutation([raw[m][s]["summary"]["symbols_at_target_selective_fb_entropy"] for s in seeds], [raw["frozen"][s]["summary"]["symbols_at_target_fixed"] for s in seeds])
    adjusted = holm([tf_p[m] for m in ("frozen", "context_only", "nested")])
    put("tf.seed_test_vs_frozen_fixed", {m: {"p": tf_p[m], "p_holm": a} for m, a in zip(("frozen", "context_only", "nested"), adjusted)}, src_raw)
    # where a curve already exceeds the target at its cheapest point, the reading is that point (an upper bound)
    floors = {}
    for m in ("frozen", "context_only", "nested"):
        floors[f"tf.{m}"] = {s: round(raw[m][s]["selective_curve"][0]["tok_acc"], 4) for s in sorted(raw[m])}
        steps = [x for s in sorted(raw[m]) for x in raw[m][s]["live"]]
        floors[f"real.{m}"] = round(sum(x["live"][0]["tok_acc"] for x in steps) / len(steps), 4)
    put("floor.accuracy_at_cheapest_point", floors, src_raw)
    # emulation check on the fixed-prefix receivers as well (real against emulated, same audited steps)
    fixed_gap = {m: 100 * (float(cost[m]["free-running fixed prefix"]) / float(cost[m]["emulated fixed prefix, same steps"]) - 1) for m in ("frozen", "context_only", "nested")}
    put("emulation_gap_fixed_pct", fixed_gap, f"{six}/tables/refine_cost_hier.csv")

    # ---- pilot ablations at full precision (raw logs of the GPU reproduction)
    import tarfile

    pilot_dir = ROOT / "runs" / "paper_pilot_raw"
    if not pilot_dir.exists():
        pilot_dir.mkdir(parents=True)
        with tarfile.open(ROOT / pilot / "raw_streams.tar.gz") as tar:
            tar.extractall(pilot_dir)
    praw: dict[str, dict[int, dict]] = {}
    for path in sorted(pilot_dir.rglob("hier_*.json")):
        run = json.load(open(path, encoding="utf-8"))
        if path.stem != f"hier_{run['method']}_s{run['seed']}":
            continue  # sensitivity variants and other channels
        praw.setdefault(run["method"], {})[run["seed"]] = run
    full = {}
    for m in sorted(praw):
        if m == "nested":
            continue
        shared = sorted(set(praw[m]) & set(praw["nested"]))
        series = [np.asarray(praw[m][s]["log"]["prior_bits"]) - np.asarray(praw["nested"][s]["log"]["prior_bits"]) for s in shared]
        ci = block_bootstrap(series, block=25)
        eq = block_equivalence(series, 0.05, block=25)
        full[m] = {"seeds": shared, "delta": ci["delta"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"], "eq90": [eq["ci_low"], eq["ci_high"]], "equivalent": eq["equivalent"]}
    put("pilot.full_precision", full, f"{pilot}/raw_streams.tar.gz")
    power_file = ROOT / "results" / "refine_power" / "refine_power.json"
    if power_file.exists():
        pw = json.load(open(power_file, encoding="utf-8"))
        put("refine_power.held_out", {v: {"power_tau_0.9": r["power_at_tau_0.9"], "fixed": r["fixed_at_target"], "message_gain": r["message_gain"]["at_target"], "subset_gain": r["subset_gain"]["at_target"], "change_pct": 100 * (r["subset_gain"]["at_target"]["total_entropy"] / r["message_gain"]["at_target"]["total_entropy"] - 1), "messages": pw["messages"], "tokens": r["tokens"]} for v, r in pw["results"].items()}, "results/refine_power/refine_power.json")
    seed0 = ROOT / "results" / "refine_power" / "stream_seed0.json"
    if seed0.exists():
        put("refine_power.stream_seed0", json.load(open(seed0, encoding="utf-8")), "results/refine_power/stream_seed0.json (scripts/compare_power.py)")
    sep = {m: float(stream[m]["prior bits/token"].split()[0]) / cap for m in ("frozen", "context_only", "nested")}
    put("link.separation_benchmark_symbols_per_token", sep, f"{six}/tables/stream_hier.csv and C(10 dB)")
    put("link.cost_over_benchmark", {"dnsc_refine": 7.0754 / sep["nested"], "frozen_fixed": 9.2272 / sep["frozen"]}, "tf readings / separation benchmark")
    put("link.levels", raw["nested"][0]["levels"], src_raw)
    put("link.slot_switches_mean", sum(raw["nested"][s]["slot_switches"] for s in raw["nested"]) / len(raw["nested"]), src_raw)
    put("runtime.hours", {f"{m}_s{s}": round(raw[m][s]["summary"]["seconds"] / 3600, 2) for m in raw for s in sorted(raw[m])}, src_raw)
    put("runtime.platform", {f"{m}_s{s}": f"{raw[m][s]['env'].get('platform', '')[:20]} / {raw[m][s]['env'].get('threads')} threads" for m in raw for s in sorted(raw[m])}, src_raw)

    (ROOT / "paper").mkdir(exist_ok=True)
    json.dump({"values": out, "sources": src}, open(ROOT / "paper" / "numbers.json", "w", encoding="utf-8"), indent=1)
    lines = ["# Numbers quoted in the manuscript", "", "Generated by `scripts/paper_numbers.py` from the result files.", "", "| Key | Value | Source |", "| --- | --- | --- |"]
    for key, value in out.items():
        shown = json.dumps(value) if not isinstance(value, float) else f"{value:.4f}"
        lines.append(f"| `{key}` | {shown[:160]} | `{src[key]}` |")
    (ROOT / "paper" / "NUMBERS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[numbers] {len(out)} values -> paper/numbers.json, paper/NUMBERS.md")


if __name__ == "__main__":
    main()
