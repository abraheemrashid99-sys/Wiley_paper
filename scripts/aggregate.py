"""Turn raw result files into the tables of the paper.

Every number in a table is computed here from the JSON files written by the
experiment scripts - nothing is typed by hand - and each table is emitted as
Markdown, CSV and LaTeX under ``<out_dir>/tables``.

    python scripts/aggregate.py --config configs/main.yaml --ref nested
    python scripts/aggregate.py --config configs/gpt2.yaml --seeds 3,4,5 --to tables_unseen

``--seeds`` restricts the stream tables to some seeds (for instance to the
seeds that no earlier study had seen) and ``--to`` names the folder they go to.
"""

from __future__ import annotations

import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from dnsc.cli import int_list, make_parser, parse
from dnsc.headroom import format_rows
from dnsc.channel import capacity_real
from dnsc.linkbudget import audit_comparison, crossover_traffic, net_semantic_rate, pooled_live, symbols_at_accuracy
from dnsc.stats import block_bootstrap, block_equivalence, holm, mean_sd, min_attainable_p, paired_permutation
from dnsc.utils import ensure_dir, load_json


def write_table(frame: pd.DataFrame, path: Path, caption: str) -> None:
    frame.to_csv(path.with_suffix(".csv"), index=False)
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    path.with_suffix(".md").write_text(f"**{caption}**\n\n" + "\n".join([header, rule, *body]) + "\n", encoding="utf-8")
    cols = "l" * len(frame.columns)
    escape = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("±", r"$\pm$")  # noqa: E731
    lines = [r"\begin{tabular}{" + cols + "}", r"\hline", " & ".join(escape(c) for c in frame.columns) + r" \\", r"\hline"]
    lines += [" & ".join(escape(v) for v in row) + r" \\" for row in frame.itertuples(index=False)]
    lines += [r"\hline", r"\end{tabular}"]
    path.with_suffix(".tex").write_text(f"% {caption}\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{caption}\n{frame.to_string(index=False)}")


def recovery_steps(result: dict, window: int = 10, fraction: float = 0.9) -> float:
    """Mean number of steps, after a slow-regime switch, until the smoothed
    prequential token accuracy regains ``fraction`` of that regime's final level."""
    acc, regime = np.asarray(result["log"]["tok_acc"]), np.asarray(result["log"]["regime"])
    out = []
    for r in np.unique(regime)[1:]:
        idx = np.where(regime == r)[0]
        if len(idx) < 2 * window:
            continue
        seg = acc[idx]
        smooth = np.convolve(seg, np.ones(window) / window, mode="valid")
        target = fraction * smooth[-window:].mean()
        reached = np.where(smooth >= target)[0]
        out.append(float(reached[0]) if len(reached) else float(len(seg)))
    return float(np.mean(out)) if out else float("nan")


def run_tag(path: str, result: dict) -> str:
    """The ``--tag`` a stream run was started with, read off its file name
    (``<stream>_<method>_s<seed>[_<tag>].json``)."""
    return Path(path).stem.split(f"_{result['method']}_s{result['seed']}", 1)[-1].lstrip("_")


def _differences(a: dict, b: dict, prefix: str = "") -> list[str]:
    """Config keys on which a variant run differs from its reference run."""
    out = []
    for key in sorted(set(a) | set(b)):
        if key in ("out_dir", "threads", "name"):
            continue
        va, vb = a.get(key), b.get(key)
        if isinstance(va, dict) and isinstance(vb, dict):
            out += _differences(va, vb, f"{prefix}{key}.")
        elif va != vb:
            out.append(f"{prefix}{key}={va}")
    return out


def _same_traffic(a: dict, b: dict) -> bool:
    """True when two runs carried the same messages over the same channel
    (identical stream and channel configuration and the same seed)."""
    return a["config"]["stream"] == b["config"]["stream"] and a["config"]["channel"] == b["config"]["channel"] and int(a["seed"]) == int(b["seed"])


def sensitivity_table(by_kind: dict, tagged: list, tables: Path) -> None:
    """Variant runs next to the run they vary (same stream, method and seed)."""
    rows = []
    by_tag = {(tag, run["stream"], run["method"], int(run["seed"])): run for tag, run in tagged}
    for tag, result in sorted(tagged, key=lambda item: (item[1]["stream"], item[1]["method"], item[0], int(item[1]["seed"]))):
        reference = by_kind.get(result["stream"], {}).get(result["method"], {}).get(int(result["seed"]))
        for label, run in (("(reference)", reference), (tag, result)):
            if run is None or (label == "(reference)" and any(r["variant"] == label and r["stream"] == run["stream"] and r["method"] == run["method"] and r["seed"] == run["seed"] for r in rows)):
                continue
            summary = run["summary"]
            need = summary["symbols_at_target_selective"]
            # traffic above which this run beats the context-only receiver on the SAME messages and channel:
            # the context-only run of the same variant if there is one, else the untagged one if it saw the same traffic
            free = by_tag.get((tag, run["stream"], "context_only", int(run["seed"]))) if label != "(reference)" else None
            if free is None:
                free = by_kind.get(run["stream"], {}).get("context_only", {}).get(int(run["seed"]))
            same_channel = free is not None and _same_traffic(free, run)
            star = "-"
            if same_channel and need is not None and free["summary"]["symbols_at_target_selective"] is not None and summary["sync_bits"] > 0:
                saved = (free["summary"]["symbols_at_target_selective"] - need) * summary["tokens_per_message"]
                value = crossover_traffic(summary["sync_bits_per_step"], saved, float(run["config"]["stream"]["snr"]["mean"]))
                star = "never" if np.isinf(value) else f"{value:.0f}"
            rows.append(
                {
                    "stream": run["stream"],
                    "method": run["method"],
                    "seed": run["seed"],
                    "variant": label,
                    "changed": "-" if label == "(reference)" or reference is None else "; ".join(_differences(run["config"], reference["config"])) or "-",
                    "prior bits/token": f"{summary['mean_prior_bits']:.4f}",
                    "tok-acc @m_op": f"{summary['mean_tok_acc']:.4f}",
                    "selective sym/token": f"{summary['sel_symbols']:.3f}",
                    "selective tok-acc": f"{summary['sel_tok_acc']:.4f}",
                    f"sym/token for {run['target_acc']:.0%}": "not reached" if need is None else f"{need:.3f}",
                    "sync kbit/step": f"{summary['sync_bits_per_step'] / 1e3:.2f}",
                    "N* (vs context_only)": star,
                    "slot switches": run.get("slot_switches", 0),
                }
            )
    if rows:
        write_table(pd.DataFrame(rows), tables / "sensitivity", "Variant runs next to their reference run (same stream kind, method and seed); N* is given only against a context-only run that carried the same messages over the same channel")


def stream_tables(cfg, out_dir: Path, tables: Path, ref: str, only_seeds: list[int] | None = None) -> None:
    files = sorted(glob.glob(str(out_dir / "stream" / "*.json")))
    by_kind: dict[str, dict[str, dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    tagged = []
    for path in files:
        result = load_json(path)
        if only_seeds and int(result["seed"]) not in only_seeds:
            continue
        tag = run_tag(path, result)
        if tag:  # a variant run (sensitivity study): kept out of the main tables
            tagged.append((tag, result))
        else:
            by_kind[result["stream"]][result["method"]][int(result["seed"])] = result
    sensitivity_table(by_kind, tagged, tables)
    metrics = [
        ("mean_tok_acc", "tok-acc @m_op"),
        ("mean_ce_bits", "bits/token @m_op"),
        ("mean_prior_bits", "prior bits/token"),
        ("sel_symbols", "selective sym/token"),
        ("sel_tok_acc", "selective tok-acc"),
        ("ece", "ECE"),
        ("final_probe_tok_acc", "retention tok-acc"),
        ("final_probe_bleu4", "retention BLEU-4"),
    ]
    for kind, methods in by_kind.items():
        rows, pvals, plabels = [], [], []
        for method, runs in sorted(methods.items()):
            seeds = sorted(runs)
            row = {"method": method, "seeds": len(seeds)}
            for key, label in metrics:
                mean, sd = mean_sd([runs[s]["summary"][key] for s in seeds])
                row[label] = f"{mean:.4f} ± {sd:.4f}"
            row["slot switches"] = f"{np.mean([runs[s].get('slot_switches', 0) for s in seeds]):.1f}"
            row["codebook power"] = f"{np.mean([runs[s]['summary']['mean_codebook_power'] for s in seeds]):.3f}"
            row["sync kbit/step"] = f"{np.mean([runs[s]['summary']['sync_bits_per_step'] for s in seeds]) / 1e3:.2f}"
            row["net rate (sym/msg)"] = f"{np.mean([runs[s]['summary']['net_rate'] for s in seeds]):.2f}"
            if ref in methods and method != ref:
                shared = sorted(set(seeds) & set(methods[ref]))
                if len(shared) >= 2:
                    a = [methods[ref][s]["summary"]["mean_prior_bits"] for s in shared]
                    b = [runs[s]["summary"]["mean_prior_bits"] for s in shared]
                    pvals.append(paired_permutation(a, b))
                    plabels.append(method)
                    row[f"Δ prior bits ({ref} − this)"] = f"{np.mean(a) - np.mean(b):+.4f}"
            rows.append(row)
        # paired per-step comparison with the reference method (same messages, same noise);
        # 25-step blocks (shorter on very short streams) pooled over seeds are resampled
        if ref in methods:
            k_tau = [p["threshold"] for p in next(iter(methods[ref].values()))["selective_curve"]].index(next(iter(methods[ref].values()))["threshold"])
            for row in rows:
                method = row["method"]
                shared = sorted(set(methods[method]) & set(methods[ref]))
                if method == ref or not shared:
                    continue
                for key, label, pick in (("prior_bits", "prior bits", None), ("sel_symbols", "selective sym/token", k_tau), ("sel_acc", "selective tok-acc", k_tau)):
                    series = []
                    for seed in shared:
                        mine, theirs = np.asarray(methods[method][seed]["log"][key], dtype=float), np.asarray(methods[ref][seed]["log"][key], dtype=float)
                        series.append((mine - theirs) if pick is None else (mine[:, pick] - theirs[:, pick]))
                    ci = block_bootstrap(series, block=max(1, min(25, len(series[0]) // 4)))
                    row[f"{label}: this − {ref} [95% CI]"] = f"{ci['delta']:+.4f} [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]" + ("*" if ci["excludes_zero"] else "")
        if pvals:
            adjusted = dict(zip(plabels, holm(pvals)))
            raw = dict(zip(plabels, pvals))
            for row in rows:
                if row["method"] in adjusted:
                    n_shared = len(set(methods[row["method"]]) & set(methods[ref]))
                    row["p (seed-level)"] = f"{raw[row['method']]:.4f}"
                    row["p (Holm)"] = f"{adjusted[row['method']]:.4f}"
                    row["smallest attainable p"] = f"{min_attainable_p(n_shared):.4f}"
                else:
                    row["p (seed-level)"] = row["p (Holm)"] = row["smallest attainable p"] = "-"
        frame = pd.DataFrame(rows).fillna("-")
        write_table(frame, tables / f"stream_{kind}", f"Online adaptation on the '{kind}' drift stream (mean ± SD over seeds; per-step paired differences to '{ref}' with a 95% block-bootstrap interval, * = excludes zero; exact seed-level sign-flip permutation test on the prior bits, raw and Holm-corrected, with the smallest p-value that number of seeds can give: a seed-level test cannot be significant at 0.05 with fewer than six seeds)")
        refinement_tables(cfg, kind, methods, tables, ref)

        # Net semantic rate against traffic volume, and the adaptation crossover.
        # Payload: the symbols per token each method needs to reach the target
        # token accuracy under selective refinement, read off its measured curve.
        snr = float(cfg["stream"]["snr"]["mean"])

        def needed(runs: dict, seeds: list | None = None) -> float:
            if seeds is not None and not set(seeds) <= set(runs):
                return float("nan")  # a reference must cover the seeds it is compared on
            values = [r["summary"]["symbols_at_target_selective"] for seed, r in runs.items() if seeds is None or seed in seeds]
            return float("nan") if any(v is None for v in values) else float(np.mean(values))

        sweep = []
        for method, runs in sorted(methods.items()):
            any_run = next(iter(runs.values()))
            # references on the same seeds as this method
            frozen_need = needed(methods["frozen"], list(runs)) if "frozen" in methods else float("nan")
            context_need = needed(methods["context_only"], list(runs)) if "context_only" in methods else float("nan")
            bits = float(np.mean([r["summary"]["sync_bits"] for r in runs.values()]))
            tokens = float(np.mean([r["summary"]["tokens_per_message"] for r in runs.values()]))
            need = needed(runs)
            row = {"method": method, f"sym/token for {any_run['target_acc']:.0%} tok-acc": "not reached" if np.isnan(need) else f"{need:.3f}"}
            if not np.isnan(need):
                if not np.isnan(frozen_need):
                    row["saving vs frozen"] = f"{1 - need / frozen_need:+.1%}"
                payload = need * tokens
                row["payload (sym/msg)"] = f"{payload:.1f}"
                for traffic in (1, 10, 100, 1000, 10000, 100000):
                    row[f"N={traffic}"] = f"{net_semantic_rate(payload, bits, any_run['n_steps'], traffic, snr):.1f}"
                # traffic per adaptation step above which shipping updates beats a receiver that ships nothing
                for label, reference in (("frozen", frozen_need), ("context_only", context_need)):
                    if bits > 0 and not np.isnan(reference):
                        star = crossover_traffic(bits / any_run["n_steps"], (reference - need) * tokens, snr)
                        row[f"N* (vs {label})"] = "never" if np.isinf(star) else f"{star:.0f}"
            sweep.append(row)
        write_table(pd.DataFrame(sweep).fillna("-"), tables / f"netrate_{kind}", f"Derived net semantic rate (real symbols per message, payload at equal fidelity plus amortised synchronisation) on the '{kind}' stream as a function of messages carried per adaptation step N, at {snr:g} dB")

        # where the gains come from: the same metrics per slow regime label (domain or era)
        per_domain = []
        for method, runs in sorted(methods.items()):
            first = next(iter(runs.values()))
            k_tau = [p["threshold"] for p in first["selective_curve"]].index(first["threshold"])
            labels = list(dict.fromkeys(first["log"]["domain"]))
            for label in labels:
                cells = defaultdict(list)
                for run in runs.values():
                    pick = np.asarray(run["log"]["domain"]) == label
                    cells["prior bits/token"].append(np.asarray(run["log"]["prior_bits"])[pick].mean())
                    cells["tok-acc @m_op"].append(np.asarray(run["log"]["tok_acc"])[pick].mean())
                    cells["selective sym/token"].append(np.asarray(run["log"]["sel_symbols"])[pick][:, k_tau].mean())
                    cells["selective tok-acc"].append(np.asarray(run["log"]["sel_acc"])[pick][:, k_tau].mean())
                row = {"method": method, "regime": label, "seeds": len(runs)}
                for key, values in cells.items():
                    mean, sd = mean_sd(values)
                    row[key] = f"{mean:.4f} ± {sd:.4f}"
                final = [run["probes"][-1]["domains"].get(label) for run in runs.values()]
                if all(final):
                    row["held-out prior bits (end)"] = f"{np.mean([f['prior_bits'] for f in final if 'prior_bits' in f]):.4f}" if all("prior_bits" in f for f in final) else "-"
                    row["held-out tok-acc (end)"] = f"{np.mean([f['tok_acc'] for f in final]):.4f}"
                recorded = all("staleness" in run["probes"][-1] for run in runs.values())
                stale = [max(run["probes"][-1]["staleness"].values()) for run in runs.values() if run["probes"][-1].get("staleness")]
                # "-": the method ships nothing; "n/a": not recorded by the code version that produced the run
                row["receiver staleness at the probe (steps)"] = f"{np.mean(stale):.0f}" if stale else ("-" if recorded else "n/a")
                per_domain.append(row)
        write_table(pd.DataFrame(per_domain).fillna("-"), tables / f"regimes_{kind}", f"The '{kind}' stream by slow regime: prequential metrics while the regime is active, and held-out probes of it at the end of the stream (mean ± SD over seeds). A probe scores the receiver's copy at that moment: methods are comparable on a probe only at equal staleness (steps since the slowest level was shipped)")

        # operating curves of every method: fixed prefix lengths and selective refinement
        curves = []
        for method, runs in sorted(methods.items()):
            row = {"method": method}
            first = next(iter(runs.values()))
            for index, point in enumerate(first["fixed_curve"]):
                row[f"acc m={point['m']}"] = f"{np.mean([r['fixed_curve'][index]['tok_acc'] for r in runs.values()]):.4f}"
            for index, point in enumerate(first["selective_curve"]):
                row[f"τ={point['threshold']:g}"] = f"{np.mean([r['selective_curve'][index]['symbols'] for r in runs.values()]):.2f} sym / {np.mean([r['selective_curve'][index]['tok_acc'] for r in runs.values()]):.4f}"
            curves.append(row)
        write_table(pd.DataFrame(curves), tables / f"curves_{kind}", f"Operating curves on the '{kind}' stream: token accuracy at every fixed prefix length, and (symbols per token / token accuracy) under selective refinement at every confidence threshold")


def _mean_or_none(values) -> float | None:
    values = list(values)
    return None if not values or any(v is None for v in values) else float(np.mean(values))


def _show(value, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def refinement_tables(cfg, kind: str, methods: dict, tables: Path, ref: str) -> None:
    """What selective refinement really costs and whether its emulation holds:
    return-link signalling, the choice of the first shell, the free-running
    audit, other target accuracies, and the equivalence tests."""
    target_default = next(iter(next(iter(methods.values())).values()))["target_acc"]

    # --- 1. forward symbols, signalling, total; emulated against free-running decoding
    # "n/a": the run was made by a code version that did not record the quantity;
    # "not reached": recorded, but the measured curve never reaches the target accuracy
    def value(summaries: list[dict], key: str) -> tuple[float | None, str]:
        if not summaries or any(key not in s for s in summaries):
            return None, "n/a"
        if any(s[key] is None for s in summaries):
            return None, "not reached"
        return float(np.mean([s[key] for s in summaries])), ""

    def cell(summaries: list[dict], key: str, digits: int = 3) -> str:
        number, why = value(summaries, key)
        return why if number is None else f"{number:.{digits}f}"

    def saving(summaries: list[dict], key: str, reference: str) -> str:
        (a, why_a), (b, why_b) = value(summaries, key), value(summaries, reference)
        return (why_a or why_b) if a is None or b is None else f"{a / b - 1:+.1%}"

    def pooled_cell(number: float | None, audit: dict | None) -> str:
        return "n/a" if audit is None else ("not reached" if number is None else f"{number:.3f}")

    def pooled_saving(a: float | None, b: float | None, audit: dict | None) -> str:
        return "n/a" if audit is None else ("not reached" if a is None or b is None else f"{a / b - 1:+.1%}")

    rows = []
    for method, runs in sorted(methods.items()):
        summaries = [r["summary"] for r in runs.values()]
        audited = [s for s in summaries if s.get("live_steps")] if all(s.get("live_steps") for s in summaries) else []
        pooled = pooled_live(list(runs.values()), target_default) if audited else None
        get = lambda key: None if pooled is None else pooled[key]  # noqa: E731,B023
        row = {
            "method": method,
            "seeds": len(runs),
            "fixed prefix": cell(summaries, "symbols_at_target_fixed"),
            "selective, forward only": cell(summaries, "symbols_at_target_selective"),
            "feedback bits/token (bitmap)": cell(summaries, "sel_feedback_bits"),
            "feedback bits/token (entropy)": cell(summaries, "sel_feedback_entropy_bits"),
            "rounds/message": cell(summaries, "sel_rounds", 2),
            "selective + feedback (bitmap)": cell(summaries, "symbols_at_target_selective_fb"),
            "selective + feedback (entropy)": cell(summaries, "symbols_at_target_selective_fb_entropy"),
            "saving vs fixed, feedback charged (entropy)": saving(summaries, "symbols_at_target_selective_fb_entropy", "symbols_at_target_fixed"),
            "free-running: steps audited": str(pooled["audited_steps"]) if pooled else "n/a",
            "free-running fixed prefix": pooled_cell(get("fixed"), pooled),
            "emulated fixed prefix, same steps": pooled_cell(get("emulated_fixed"), pooled),
            "free-running, forward only": pooled_cell(get("forward"), pooled),
            "emulated on the same steps": pooled_cell(get("emulated_forward"), pooled),
            "free-running + feedback (entropy)": pooled_cell(get("total_entropy"), pooled),
            "free-running + feedback (bitmap)": pooled_cell(get("total_bitmap"), pooled),
            "free-running saving vs free-running fixed, feedback charged (entropy)": pooled_saving(get("total_entropy"), get("fixed"), pooled),
            "free-running tok-acc at the threshold": cell(audited, "live_sel_tok_acc", 4),
            "emulated tok-acc at the threshold": cell(audited, "live_emulated_sel_tok_acc", 4),
        }
        rows.append(row)
    write_table(
        pd.DataFrame(rows),
        tables / f"refine_cost_{kind}",
        f"What selective refinement costs on the '{kind}' stream: real symbols per token for {target_default:.0%} token accuracy. 'Feedback' is the return-link signalling of the requests at the operating threshold, converted to symbols at the capacity of the mean SNR (derived, optimistic; bitmap = one bit per open token and round, entropy = ideal coding of those bits). Teacher-forced columns: mean over seeds of each run's reading. 'Free-running' decodes the same received symbols for real on a subset of the steps - the selective policy and the fixed prefixes alike - and 'emulated' is the teacher-forced figure on exactly those steps; these columns are read off the operating curves pooled over the seeds (all audited steps), and the free-running saving compares like with like. n/a: not recorded by the code version that produced the run; not reached: the measured curve never reaches the target accuracy",
    )

    # --- 1b. the primary endpoint decoded for real, with intervals and the seed-level test
    if "frozen" in methods:
        rows, tested = [], []
        for method, runs in sorted(methods.items(), key=lambda item: (item[0] != "frozen", item[0])):
            out = audit_comparison(runs, methods["frozen"], target_default, seed=20261015)
            if out is None:
                continue

            def interval(entry: dict) -> str:
                if entry["estimate"] is None:
                    return "not reached"
                high = "unbounded" if entry["ci_high"] is None else f"{entry['ci_high']:+.1%}"
                return f"{entry['estimate']:+.1%} [{entry['ci_low']:+.1%}, {high}]"

            mine, theirs = out["receiver"], out["reference"]
            per_seed = [(v["receiver"]["total_entropy"], v["reference"]["fixed"]) for v in out["per_seed"].values()]
            complete = all(a is not None and b is not None for a, b in per_seed)
            p_raw = paired_permutation([a for a, _ in per_seed], [b for _, b in per_seed]) if complete and len(per_seed) > 1 else None
            if p_raw is not None:
                tested.append((len(rows), p_raw))
            worst = max(out[key]["missed"] for key in ("vs_reference_entropy", "vs_reference_bitmap", "vs_own_fixed_entropy"))
            rows.append(
                {
                    "method": method,
                    "seeds": len(out["seeds"]),
                    "audited steps": out["audited_steps"],
                    "fixed prefix": _show(mine["fixed"]),
                    "fixed prefix vs frozen fixed": "—" if method == "frozen" else interval(out["vs_reference_fixed"]),
                    "selective + feedback (entropy)": _show(mine["total_entropy"]),
                    "vs frozen fixed [95% CI]": interval(out["vs_reference_entropy"]),
                    "vs own fixed (what refinement adds)": interval(out["vs_own_fixed_entropy"]),
                    "selective + feedback (bitmap)": _show(mine["total_bitmap"]),
                    "bitmap vs frozen fixed": interval(out["vs_reference_bitmap"]),
                    "draws missing the target": f"{worst:.1%}",
                    "per-seed saving vs frozen fixed (entropy)": " / ".join("not reached" if a is None or b is None else f"{a / b - 1:+.1%}" for a, b in per_seed),
                    "p (seed-level)": "-" if p_raw is None else f"{p_raw:.4f}",
                    "p (Holm)": "-",
                    "smallest attainable p": f"{min_attainable_p(len(per_seed)):.4f}",
                }
            )
        if tested:
            for (index, _), adjusted in zip(tested, holm([p for _, p in tested])):
                rows[index]["p (Holm)"] = f"{adjusted:.4f}"
        if rows:
            write_table(
                pd.DataFrame(rows),
                tables / f"realdecode_{kind}",
                f"Primary endpoint decoded for real on the '{kind}' stream: real symbols per token for {target_default:.0%} token accuracy on the audited steps, every receiver and the frozen receiver decoded free-running from the same received symbols, operating curves pooled over the seeds, selective refinement charged its return-link signalling. Relative changes with a 95% percentile interval from 2000 paired bootstrap draws of the audited steps within each seed; a draw in which a receiver's curve misses the target counts as an infinite cost ('unbounded' if that happens in more than 2.5% of the draws). Seed-level test: exact sign-flip permutation test of the per-seed selective cost with signalling (entropy) against the frozen fixed prefix of the same seed, Holm-corrected over the receivers, possible only if every seed reaches the target",
            )

    # --- 2. the first shell of the refinement (fewer rounds, less signalling, more forward symbols)
    rows = []
    for method, runs in sorted(methods.items()):
        grids = [r.get("refine_grid") for r in runs.values()]
        if not all(grids):
            continue
        for index, entry in enumerate(grids[0]):
            cells = [g[index] for g in grids]
            rows.append(
                {
                    "method": method,
                    "seeds": len(runs),
                    "first shell (symbols)": entry["first_m"],
                    "forward only": _show(_mean_or_none(c["symbols_at_target"] for c in cells)),
                    "+ feedback (bitmap)": _show(_mean_or_none(c["symbols_at_target_fb"] for c in cells)),
                    "+ feedback (entropy)": _show(_mean_or_none(c["symbols_at_target_fb_entropy"] for c in cells)),
                }
            )
    if rows:
        write_table(pd.DataFrame(rows), tables / f"refine_start_{kind}", f"Selective refinement on the '{kind}' stream by the shell sent in the first round: symbols per token for {target_default:.0%} token accuracy (n/a = the target is not reached on that curve). The paper's policy is the first row of every method; the other rows are a post-hoc exploration, not a tuned result")

    # --- 3. other target accuracies, read off the same measured curves
    rows = []
    for method, runs in sorted(methods.items()):
        row = {"method": method, "seeds": len(runs)}
        for target in (0.85, 0.90, 0.95):
            fixed = _mean_or_none((lambda v: None if not np.isfinite(v) else v)(symbols_at_accuracy([float(p["m"]) for p in r["fixed_curve"]], [p["tok_acc"] for p in r["fixed_curve"]], target)) for r in runs.values())
            selective = _mean_or_none((lambda v: None if not np.isfinite(v) else v)(symbols_at_accuracy([p["symbols"] for p in r["selective_curve"]], [p["tok_acc"] for p in r["selective_curve"]], target)) for r in runs.values())
            row[f"fixed, {target:.0%}"] = _show(fixed)
            row[f"selective, {target:.0%}"] = _show(selective)
        rows.append(row)
    write_table(pd.DataFrame(rows), tables / f"targets_{kind}", f"Forward symbols per token needed on the '{kind}' stream for three target token accuracies (mean over seeds; n/a = not reached within the measured curve, or below its cheapest point for one of the seeds)")

    # --- 4. equivalence ("no meaningful difference") with margins fixed in the config
    if ref in methods:
        margins = {"prior_bits": float(cfg["stats"]["equiv_margin_bits"]), "sel_symbols": float(cfg["stats"]["equiv_margin_symbols"])}
        first = next(iter(methods[ref].values()))
        k_tau = [p["threshold"] for p in first["selective_curve"]].index(first["threshold"])
        rows = []
        for method in sorted(methods):
            shared = sorted(set(methods[method]) & set(methods[ref]))
            if method == ref or not shared:
                continue
            row = {"method": method, "seeds": len(shared)}
            for key, label, pick in (("prior_bits", "prior bits/token", None), ("sel_symbols", "selective sym/token", k_tau)):
                series = []
                for seed in shared:
                    mine, theirs = np.asarray(methods[method][seed]["log"][key], dtype=float), np.asarray(methods[ref][seed]["log"][key], dtype=float)
                    series.append((mine - theirs) if pick is None else (mine[:, pick] - theirs[:, pick]))
                test = block_equivalence(series, margins[key], block=max(1, min(25, len(series[0]) // 4)))
                row[f"{label}: this − {ref} [90% CI]"] = f"{test['delta']:+.4f} [{test['ci_low']:+.4f}, {test['ci_high']:+.4f}]"
                row[f"{label}: equivalent within ±{margins[key]:g}"] = "yes" if test["equivalent"] else "no"
            rows.append(row)
        if rows:
            write_table(pd.DataFrame(rows), tables / f"equivalence_{kind}", f"Equivalence tests against '{ref}' on the '{kind}' stream: two one-sided tests at 5% on the per-step paired differences (90% block-bootstrap interval inside the margin). The margins are fixed in the configuration before the runs; 'no' means equivalence was not shown, not that a difference was shown")


def cost_table(out_dir: Path, tables: Path) -> None:
    """Measured computational cost on the machine that ran the benchmark."""
    path = out_dir / "eval" / "speed.json"
    if not path.exists():
        return
    speed = load_json(path)
    rows = [
        {"quantity": "device", "value": f"{speed['device']} ({speed['threads']} CPU threads)"},
        {"quantity": "receiver, forward only (stream step, it/s)", "value": f"{speed['stream_forward_it_per_s']:.2f}"},
        {"quantity": "transmitter, bias adaptation (stream step, it/s)", "value": f"{speed['stream_medium_it_per_s']:.2f}"},
        {"quantity": "transmitter, bias + LoRA adaptation (stream step, it/s)", "value": f"{speed['stream_full_it_per_s']:.2f}"},
        {"quantity": "codebook training (it/s)", "value": f"{speed['pretrain_it_per_s']:.2f}"},
    ]
    for key, value in speed.get("parameters", {}).items():
        rows.append({"quantity": f"parameters: {key}", "value": f"{value:,}"})
    if speed.get("peak_rss_mb"):
        rows.append({"quantity": "peak resident memory of the benchmark process (MB)", "value": f"{speed['peak_rss_mb']:.0f}"})
    for key, value in sorted(speed["decode_ms"].items(), key=lambda kv: (kv[0].split("_m")[0], int(kv[0].split("_m")[1]))):
        prior, m = key.split("_m")
        rows.append({"quantity": f"decode latency, one message of {speed['tokens']} tokens, prior '{prior}', m={m} (ms)", "value": f"{value:.1f}"})
    write_table(pd.DataFrame(rows), tables / "cost", "Measured computational cost (scripts/benchmark_speed.py; wall-clock on the machine named in the first row, not extrapolated to any other device; energy was not measured)")


def rate_tables(out_dir: Path, tables: Path) -> None:
    for path in sorted(glob.glob(str(out_dir / "eval" / "rate_*.json"))):
        data = load_json(path)
        frame = pd.DataFrame(data["rows"])
        codec_rows = frame[frame["system"].str.startswith("dnsc_")]
        for metric in ("bleu4", "sim", "chrf", "wer", "gen_tok_acc", "tok_acc"):
            if metric not in codec_rows.columns:
                continue
            sub = codec_rows.dropna(subset=[metric])
            if sub.empty:
                continue
            pivot = sub.pivot_table(index=["channel", "prior", "m"], columns="snr_db", values=metric).round(4).reset_index()
            pivot.columns = [str(c) if not isinstance(c, float) else f"{c:g} dB" for c in pivot.columns]
            write_table(pivot, tables / f"rate_{data['corpus']}_{metric}", f"Semantic codec, {metric} on {data['corpus']} (n={data['n']}, {data['mean_tokens']:.1f} tokens per sentence) by decoder prior, symbols per token m and SNR")
        power = codec_rows[codec_rows["prior"] == codec_rows["prior"].iloc[0]].groupby("m")["codebook_power"].mean().round(4).reset_index()
        write_table(power, tables / f"rate_{data['corpus']}_power", f"Average codeword power per symbol on {data['corpus']} before block power control (the transmitted power is exactly 1)")
        ref = frame[~frame["system"].str.startswith("dnsc_")]
        if not ref.empty:
            pivot = ref.pivot_table(index=["system", "channel", "m"], columns="snr_db", values="delivered").round(4).reset_index()
            pivot.columns = [str(c) if not isinstance(c, float) else f"{c:g} dB" for c in pivot.columns]
            write_table(pivot, tables / f"rate_{data['corpus']}_separation", f"Derived: fraction of messages an ideal separate source-channel design delivers on {data['corpus']} at the same symbol budget")


def refine_tables(out_dir: Path, tables: Path) -> None:
    for path in sorted(glob.glob(str(out_dir / "eval" / "refine_*.json"))):
        data = load_json(path)
        rows, live = [], []
        for res in data["results"]:
            top = max(res["fixed"], key=lambda r: r["quality"])
            # cheapest selective point within 0.005 token accuracy of the best fixed rate
            good = [c for c in res["curve"] if c["quality"] >= top["quality"] - 0.005]
            pick = min(good, key=lambda c: c["symbols"]) if good else max(res["curve"], key=lambda c: c["quality"])
            best_gain = max(res["curve"], key=lambda c: c["gain_over_fixed"])
            rows.append(
                {
                    "channel": res["channel"],
                    "snr_db": res["snr_db"],
                    "ECE raw": f"{res['ece_raw']:.3f}",
                    "ECE calibrated": f"{res['ece_calibrated']:.3f}",
                    "best fixed m": top["m"],
                    "fixed acc": f"{top['quality']:.3f}",
                    "selective sym/token": f"{pick['symbols']:.2f}",
                    "selective acc": f"{pick['quality']:.3f}",
                    "saving": f"{1 - pick['symbols'] / top['symbols']:.1%}",
                    "feedback bits/token (bitmap / entropy)": f"{pick['feedback_bits']:.2f} / {pick['feedback_entropy_bits']:.2f}" if "feedback_bits" in pick else "n/a",
                    "saving, feedback charged (entropy)": f"{1 - (pick['symbols'] + pick['feedback_entropy_bits'] / capacity_real(res['snr_db'])) / top['symbols']:.1%}" if "feedback_bits" in pick else "n/a",
                    "max gain vs fixed": f"{best_gain['gain_over_fixed']:+.3f} @ {best_gain['symbols']:.2f}",
                    "oracle sym/token": f"{res['oracle']['symbols']:.2f}",
                    "random control acc": f"{res['random_matched_at_mid_threshold']['quality']:.3f}",
                }
            )
            for point in res.get("free_running", []):
                live.append(
                    {
                        "channel": res["channel"],
                        "snr_db": res["snr_db"],
                        "threshold": f"{point['threshold']:.3f}",
                        "sym/token": f"{point['symbols']:.2f}",
                        "token acc": f"{point['gen_tok_acc']:.3f}",
                        "BLEU-4": f"{point['bleu4']:.3f}",
                        "requests/token": f"{point['requests_per_token']:.3f}",
                        "feedback bits/token (bitmap / entropy)": f"{point['feedback_bits']:.2f} / {point['feedback_entropy_bits']:.2f}" if "feedback_bits" in point else "n/a",
                        "sym/token + feedback (entropy)": f"{point['symbols'] + point['feedback_entropy_bits'] / capacity_real(res['snr_db']):.2f}" if "feedback_bits" in point else "n/a",
                    }
                )
        write_table(pd.DataFrame(rows), tables / f"refine_{data['corpus']}", f"Selective refinement on {data['corpus']}, teacher-forced sweep (token accuracy; ECE of token confidence)")
        if live:
            write_table(pd.DataFrame(live), tables / f"refine_{data['corpus']}_live", f"Selective refinement on {data['corpus']}, free-running multi-round decoding")


def headroom_table(out_dir: Path, tables: Path) -> None:
    path = out_dir / "eval" / "headroom.json"
    if not path.exists():
        return
    result = load_json(path)
    rows = format_rows(result)
    if rows:
        write_table(pd.DataFrame(rows), tables / "headroom", f"Diagnostic: what each memory could buy with oracle inputs, at m={result['m_op']} symbols/token and {result['snr_db']:g} dB ({result['steps']} adaptation steps; negative Δ bits = better)")


def decoder_table(out_dir: Path, tables: Path) -> None:
    """The selection of the refinement decoder on validation text (scripts/tune_decoder.py)."""
    path = out_dir / "eval" / "decoder_selection.json"
    if not path.exists():
        return
    result = load_json(path)
    decision = result.get("decision")
    if not decision:
        return
    show = lambda v: "not reached" if v is None else f"{v:.3f}"  # noqa: E731
    rows = []
    for name, c in sorted(result["candidates"].items(), key=lambda item: (item[1]["beam"], item[1]["start_m"], item[1]["reopen"])):
        at = {p["threshold"]: p for p in c["points"]}
        rows.append(
            {
                "candidate": name + (" (chosen)" if name == decision["chosen"] else ""),
                "beam": c["beam"],
                "first shell (symbols)": c["start_m"],
                "reopening": "yes" if c["reopen"] else "no",
                "forward only": show(c["forward"]),
                "+ signalling (entropy)": show(c["total_entropy"]),
                "+ signalling (bitmap)": show(c["total_bitmap"]),
                "fixed prefix, same beam": show(c["fixed"]),
                "vs fixed prefix (entropy)": "-" if c["total_entropy"] is None or c["fixed"] is None else f"{c['total_entropy'] / c['fixed'] - 1:+.1%}",
                "rounds/message at 0.95": f"{at[0.95]['rounds']:.2f}" if 0.95 in at else "-",
                "tok-acc at 0.95": f"{at[0.95]['tok_acc']:.3f}" if 0.95 in at else "-",
            }
        )
    write_table(
        pd.DataFrame(rows),
        tables / "decoder_selection",
        f"Decoder of selective refinement, chosen on validation text only ({result.get('n_messages', '?')} messages, {result.get('n_tokens', '?')} tokens; SNR mean {result.get('snr_mean', float('nan')):.1f} dB): real symbols per token for {result.get('target_acc', 0.9):.0%} token accuracy, decoded free-running, the signalling of the requests charged at its entropy (criterion) or as a bitmap; the fixed prefix is decoded with the same beam. Rule fixed in docs/METHOD.md, section 2.5: lowest criterion among the greedy candidates (ties within 1% go to fewer rounds), then a beam only if it lowers the criterion by more than 2%",
    )


def main() -> None:
    parser = make_parser(__doc__)
    parser.add_argument("--ref", default="nested", help="reference method for paired tests")
    parser.add_argument("--seeds", default="", help="restrict the stream tables to these seeds (comma-separated; default: all)")
    parser.add_argument("--to", default="tables", help="folder under the run directory for the tables")
    cfg, args = parse(parser)
    out_dir = Path(cfg["out_dir"])
    tables = ensure_dir(out_dir / args.to)
    rate_tables(out_dir, tables)
    stream_tables(cfg, out_dir, tables, args.ref, int_list(args.seeds) or None)
    refine_tables(out_dir, tables)
    headroom_table(out_dir, tables)
    decoder_table(out_dir, tables)
    cost_table(out_dir, tables)
    print(f"\n[aggregate] tables written to {tables}")


if __name__ == "__main__":
    main()
