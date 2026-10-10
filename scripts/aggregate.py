"""Turn raw result files into the tables of the paper.

Every number in a table is computed here from the JSON files written by the
experiment scripts - nothing is typed by hand - and each table is emitted as
Markdown, CSV and LaTeX under ``<out_dir>/tables``.

    python scripts/aggregate.py --config configs/main.yaml --ref nested
"""

from __future__ import annotations

import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from dnsc.cli import make_parser, parse
from dnsc.headroom import format_rows
from dnsc.linkbudget import crossover_traffic, net_semantic_rate
from dnsc.stats import holm, mean_sd, paired_permutation
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


def stream_tables(cfg, out_dir: Path, tables: Path, ref: str) -> None:
    files = sorted(glob.glob(str(out_dir / "stream" / "*.json")))
    by_kind: dict[str, dict[str, dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    for path in files:
        result = load_json(path)
        by_kind[result["stream"]][result["method"]][int(result["seed"])] = result
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
            row["recovery steps"] = f"{np.nanmean([recovery_steps(runs[s]) for s in seeds]):.1f}"
            row["codebook power"] = f"{np.mean([runs[s]['summary']['mean_codebook_power'] for s in seeds]):.3f}"
            row["sync kbit/step"] = f"{np.mean([runs[s]['summary']['sync_bits_per_step'] for s in seeds]) / 1e3:.2f}"
            row["net rate (sym/msg)"] = f"{np.mean([runs[s]['summary']['net_rate'] for s in seeds]):.2f}"
            if ref in methods and method != ref:
                shared = sorted(set(seeds) & set(methods[ref]))
                if len(shared) >= 2:
                    a = [methods[ref][s]["summary"]["mean_tok_acc"] for s in shared]
                    b = [runs[s]["summary"]["mean_tok_acc"] for s in shared]
                    pvals.append(paired_permutation(a, b))
                    plabels.append(method)
                    row[f"Δ tok-acc ({ref} − this)"] = f"{np.mean(a) - np.mean(b):+.4f}"
            rows.append(row)
        if pvals:
            adjusted = dict(zip(plabels, holm(pvals)))
            for row in rows:
                row["p (Holm)"] = f"{adjusted[row['method']]:.4f}" if row["method"] in adjusted else "-"
        frame = pd.DataFrame(rows).fillna("-")
        write_table(frame, tables / f"stream_{kind}", f"Online adaptation on the '{kind}' drift stream (mean ± SD over seeds; paired sign-flip permutation test vs '{ref}', Holm-corrected)")

        # Net semantic rate against traffic volume, and the adaptation crossover.
        # Payload: the symbols per token each method needs to reach the target
        # token accuracy under selective refinement, read off its measured curve.
        snr = float(cfg["stream"]["snr"]["mean"])

        def needed(runs: dict) -> float:
            values = [r["summary"]["symbols_at_target_selective"] for r in runs.values()]
            return float("nan") if any(v is None for v in values) else float(np.mean(values))

        frozen_need = needed(methods["frozen"]) if "frozen" in methods else float("nan")
        sweep = []
        for method, runs in sorted(methods.items()):
            any_run = next(iter(runs.values()))
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
                if bits > 0 and not np.isnan(frozen_need):  # traffic above which adapting beats not adapting
                    saved = (frozen_need - need) * tokens
                    star = crossover_traffic(bits / any_run["n_steps"], saved, snr)
                    row["N* (vs frozen)"] = "never" if np.isinf(star) else f"{star:.0f}"
            sweep.append(row)
        write_table(pd.DataFrame(sweep).fillna("-"), tables / f"netrate_{kind}", f"Derived net semantic rate (real symbols per message, payload at equal fidelity plus amortised synchronisation) on the '{kind}' stream as a function of messages carried per adaptation step N, at {snr:g} dB")

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


def rate_tables(out_dir: Path, tables: Path) -> None:
    for path in sorted(glob.glob(str(out_dir / "eval" / "rate_*.json"))):
        data = load_json(path)
        frame = pd.DataFrame(data["rows"])
        codec_rows = frame[frame["system"].str.startswith("dnsc_")]
        for metric in ("bleu4", "sim", "gen_tok_acc", "tok_acc"):
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


def main() -> None:
    parser = make_parser(__doc__)
    parser.add_argument("--ref", default="nested", help="reference method for paired tests")
    cfg, args = parse(parser)
    out_dir = Path(cfg["out_dir"])
    tables = ensure_dir(out_dir / "tables")
    rate_tables(out_dir, tables)
    stream_tables(cfg, out_dir, tables, args.ref)
    refine_tables(out_dir, tables)
    headroom_table(out_dir, tables)
    print(f"\n[aggregate] tables written to {tables}")


if __name__ == "__main__":
    main()
