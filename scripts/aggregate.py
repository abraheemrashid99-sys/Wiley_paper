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
    metrics = [("mean_tok_acc", "stream tok-acc"), ("mean_ce_bits", "stream bits/token"), ("final_probe_tok_acc", "retention tok-acc"), ("final_probe_bleu1", "retention BLEU-1"), ("final_probe_bleu4", "retention BLEU-4")]
    for kind, methods in by_kind.items():
        rows, pvals, plabels = [], [], []
        for method, runs in sorted(methods.items()):
            seeds = sorted(runs)
            row = {"method": method, "seeds": len(seeds)}
            for key, label in metrics:
                mean, sd = mean_sd([runs[s]["summary"][key] for s in seeds])
                row[label] = f"{mean:.4f} ± {sd:.4f}"
            row["recovery steps"] = f"{np.nanmean([recovery_steps(runs[s]) for s in seeds]):.1f}"
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

        # net semantic rate against traffic volume, and the adaptation crossover
        snr = float(cfg["stream"]["snr"]["mean"])
        sweep = []
        for method, runs in sorted(methods.items()):
            any_run = next(iter(runs.values()))
            bits = float(np.mean([r["summary"]["sync_bits"] for r in runs.values()]))
            row = {"method": method}
            for traffic in (1, 10, 100, 1000, 10000, 100000):
                row[f"N={traffic}"] = f"{net_semantic_rate(any_run['m_op'], bits, any_run['n_steps'], traffic, snr):.1f}"
            grans = any_run["config"]["codec"]["granularities"]
            m_index = grans.index(any_run["m_op"])
            if m_index + 1 < len(grans) and bits > 0:  # traffic at which adapting beats sending the next longer prefix
                row["N* (vs next prefix)"] = f"{crossover_traffic(bits / any_run['n_steps'], grans[m_index + 1] - grans[m_index], snr):.1f}"
            sweep.append(row)
        write_table(pd.DataFrame(sweep).fillna("-"), tables / f"netrate_{kind}", f"Derived net semantic rate (real symbols per message) on the '{kind}' stream as a function of messages carried per adaptation step N, at {snr:g} dB")


def rate_tables(out_dir: Path, tables: Path) -> None:
    for path in sorted(glob.glob(str(out_dir / "eval" / "rate_*.json"))):
        data = load_json(path)
        frame = pd.DataFrame(data["rows"])
        for metric in ("bleu4", "sim", "tok_acc"):
            if metric not in frame.columns:
                continue
            sub = frame[frame["system"] == "dnsc"].dropna(subset=[metric])
            if sub.empty:
                continue
            pivot = sub.pivot_table(index=["channel", "m"], columns="snr_db", values=metric).round(4).reset_index()
            pivot.columns = [str(c) if not isinstance(c, float) else f"{c:g} dB" for c in pivot.columns]
            write_table(pivot, tables / f"rate_{data['corpus']}_{metric}", f"Semantic codec, {metric} on {data['corpus']} (n={data['n']}) by prefix length m and SNR")
        ref = frame[frame["system"] != "dnsc"]
        if not ref.empty:
            pivot = ref.pivot_table(index=["system", "channel", "m"], columns="snr_db", values="delivered").round(4).reset_index()
            pivot.columns = [str(c) if not isinstance(c, float) else f"{c:g} dB" for c in pivot.columns]
            write_table(pivot, tables / f"rate_{data['corpus']}_separation", f"Derived: fraction of messages an ideal separate source-channel design delivers on {data['corpus']} at the same symbol budget")


def refine_tables(out_dir: Path, tables: Path) -> None:
    for path in sorted(glob.glob(str(out_dir / "eval" / "refine_*.json"))):
        data = load_json(path)
        rows = []
        for res in data["results"]:
            best = max(res["fixed"], key=lambda r: r["quality"])
            # cheapest calibrated-stopping point that is within 0.01 of the best fixed-rate quality
            good = [c for c in res["curve"] if c["quality"] >= best["quality"] - 0.01]
            pick = min(good, key=lambda c: c["symbols"]) if good else max(res["curve"], key=lambda c: c["quality"])
            rows.append(
                {
                    "channel": res["channel"],
                    "snr_db": res["snr_db"],
                    "ECE raw": f"{res['ece_raw']:.3f}",
                    "ECE calibrated": f"{res['ece_calibrated']:.3f}",
                    "best fixed m": best["m"],
                    "fixed quality": f"{best['quality']:.3f}",
                    "adaptive symbols": f"{pick['symbols']:.1f}",
                    "adaptive quality": f"{pick['quality']:.3f}",
                    "saving": f"{1 - pick['symbols'] / best['symbols']:.1%}",
                    "oracle symbols": f"{res['oracle']['symbols']:.1f}",
                }
            )
        write_table(pd.DataFrame(rows), tables / f"refine_{data['corpus']}", f"Uncertainty-triggered incremental refinement on {data['corpus']} (quality = {data['results'][0]['quality_metric']})")


def main() -> None:
    parser = make_parser(__doc__)
    parser.add_argument("--ref", default="nested", help="reference method for paired tests")
    cfg, args = parse(parser)
    out_dir = Path(cfg["out_dir"])
    tables = ensure_dir(out_dir / "tables")
    rate_tables(out_dir, tables)
    stream_tables(cfg, out_dir, tables, args.ref)
    refine_tables(out_dir, tables)
    print(f"\n[aggregate] tables written to {tables}")


if __name__ == "__main__":
    main()
