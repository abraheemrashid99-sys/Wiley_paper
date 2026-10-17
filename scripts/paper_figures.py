"""Data figures of the manuscript, drawn from the result files (nothing typed by hand).

    python scripts/paper_figures.py --out paper/figures

fig_rate_priors      free-running token accuracy against symbols per token on the
                     base corpus, by receiver prior, at 0, 10 and 20 dB (results/gpt2)
fig_operating        operating curves decoded for real on the audited steps, frozen
                     receiver and DN-SC, six seeds pooled (results/gpt2_v0.4_six_seeds)
fig_prior_stream     prior cross-entropy along the hierarchical stream, mean over seeds
fig_netrate          derived net semantic rate against messages per adaptation step
fig_primary_endpoint see scripts/paper_figure_endpoint.py (same style)
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
SIX = ROOT / "results" / "gpt2_v0.4_six_seeds"


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "font.family": "DejaVu Sans",
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK,
            "ytick.labelcolor": INK,
            "legend.frameon": False,
            "savefig.facecolor": "white",
        }
    )


def tidy(ax) -> None:
    ax.grid(color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, out: Path, name: str) -> None:
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"{name}.{suffix}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[figure] {out / name}.pdf")


def runs_by_method() -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = {}
    for path in sorted(glob.glob(str(SIX / "raw" / "hier_*.json"))) or sorted(glob.glob(str(ROOT / "runs" / "six" / "stream" / "hier_*.json"))):
        run = json.load(open(path, encoding="utf-8"))
        out.setdefault(run["method"], {})[run["seed"]] = run
    return out


# ---------------------------------------------------------------- base corpus
def fig_rate_priors(out: Path) -> None:
    with open(ROOT / "results/gpt2/tables/rate_europarl_nltk_gen_tok_acc.csv", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["channel"] == "awgn"]
    priors = [("llm_ctx", "LLM prior + true history (diagnostic)", BLUE, "o", (0, (1, 1.5))), ("llm", "LLM prior", BLUE, "o", "-"), ("unigram", "Unigram prior", ORANGE, "s", "-"), ("none", "No prior (ML)", AQUA, "^", "-")]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.45), sharey=True)
    for ax, snr in zip(axes, ("0", "10", "20")):
        for key, label, color, marker, ls in priors:
            pts = sorted((int(r["m"]), float(r[f"{snr} dB"])) for r in rows if r["prior"] == key)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], ls=ls, color=color, marker=marker, ms=4.5, lw=1.6, mec="white", mew=0.8, label=label, mfc=color if key != "llm_ctx" else "white")
        ax.set_title(f"AWGN, {snr} dB", fontsize=8.5, color=INK)
        ax.set_xticks([2, 4, 6, 8, 12, 16])
        ax.set_xlabel("Symbols per token, $m$")
        ax.set_ylim(-0.02, 1.02)
        tidy(ax)
    axes[0].set_ylabel("Token accuracy (free-running)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=7.5)
    fig.tight_layout()
    save(fig, out, "fig_rate_priors")


# ---------------------------------------------------------------- operating curves
def pooled_curves(runs: dict[int, dict]) -> tuple[list[float], list[float], list[float], list[float]]:
    seeds = sorted(runs)
    fixed_m = [float(p["m"]) for p in runs[seeds[0]]["live_fixed_curve"]]
    fixed_acc = [float(np.mean([runs[s]["live_fixed_curve"][k]["tok_acc"] for s in seeds])) for k in range(len(fixed_m))]
    n = len(runs[seeds[0]]["live_curve"])
    sel_sym = [float(np.mean([runs[s]["live_curve"][j]["symbols_total_entropy"] for s in seeds])) for j in range(n)]
    sel_acc = [float(np.mean([runs[s]["live_curve"][j]["tok_acc"] for s in seeds])) for j in range(n)]
    return fixed_m, fixed_acc, sel_sym, sel_acc


def fig_operating(out: Path, runs: dict[str, dict[int, dict]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, (method, title) in zip(axes, (("frozen", "Frozen receiver"), ("nested", "DN-SC (all memories)"))):
        fm, fa, ss, sa = pooled_curves(runs[method])
        ax.plot(fm, fa, "-", color=ORANGE, marker="s", ms=4.5, lw=1.6, mec="white", mew=0.8, label="Fixed prefix")
        ax.plot(ss, sa, "-", color=BLUE, marker="o", ms=4.5, lw=1.6, mec="white", mew=0.8, label="Selective refinement + signaling")
        ax.axhline(0.9, color=MUTED, lw=0.9, ls=(0, (4, 3)))
        ax.text(1.3, 0.905, "target 0.90", va="bottom", ha="left", fontsize=7, color=MUTED)
        ax.set_title(f"{title}, {len(runs[method])} seeds", fontsize=8.5, color=INK)
        ax.set_xlabel("Real channel symbols per token")
        ax.set_xlim(1, 16.5)
        ax.set_ylim(0, 1.02)
        tidy(ax)
    axes[0].set_ylabel("Token accuracy (decoded for real)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7.5)
    fig.tight_layout()
    save(fig, out, "fig_operating")


# ---------------------------------------------------------------- prior along the stream
def fig_prior_stream(out: Path, runs: dict[str, dict[int, dict]]) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    window = 25
    first = next(iter(runs["frozen"].values()))
    steps = np.arange(1, len(first["log"]["prior_bits"]) + 1)
    domains = first["log"]["domain"]
    # regime bands
    start = 0
    for i in range(1, len(domains) + 1):
        if i == len(domains) or domains[i] != domains[start]:
            if domains[start] == "teleqna":
                ax.axvspan(start + 0.5, i + 0.5, color="#f1f0ec", lw=0)
            ax.text((start + i) / 2, 7.55, "TeleQnA" if domains[start] == "teleqna" else "AG News", ha="center", va="center", fontsize=7, color=MUTED)
            start = i
    common = sorted(set.intersection(*(set(runs[m]) for m in ("frozen", "context_only", "nested"))))  # matched seeds
    for method, label, color in (("frozen", "Frozen receiver", ORANGE), ("context_only", "Context memory", AQUA), ("nested", "DN-SC (all memories)", BLUE)):
        seeds = common
        series = np.mean([runs[method][s]["log"]["prior_bits"] for s in seeds], axis=0)
        smooth = np.convolve(series, np.ones(window) / window, mode="valid")
        x = steps[window - 1 :]
        ax.plot(x, smooth, color=color, lw=1.6, label=label)
    ax.set_xlim(1, len(steps))
    ax.set_ylim(3.5, 7.8)
    ax.set_xlabel("Adaptation step (six messages per step)")
    ax.set_ylabel("Prior cross-entropy (bits/token)")
    tidy(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7.5)
    fig.tight_layout()
    save(fig, out, "fig_prior_stream")


# ---------------------------------------------------------------- net semantic rate
def capacity(snr_db: float) -> float:
    return 0.5 * math.log2(1 + 10 ** (snr_db / 10))


def fig_netrate(out: Path, runs: dict[str, dict[int, dict]]) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 2.9))
    cap = capacity(10.0)
    n = np.logspace(1, 5, 400)
    common = sorted(set.intersection(*(set(runs[m]) for m in ("frozen", "context_only", "nested"))))  # matched seeds
    payload, bits = {}, {}
    for method in ("frozen", "context_only", "nested"):  # payload per message, S*T of every seed averaged (as scripts/paper_numbers.py)
        payload[method] = float(np.mean([runs[method][x]["summary"]["symbols_at_target_selective_fb_entropy"] * runs[method][x]["summary"]["tokens_per_message"] for x in common]))
        bits[method] = np.mean([runs[method][x]["summary"]["sync_bits_per_step"] for x in common])
    payload["frozen_fixed"] = float(np.mean([runs["frozen"][x]["summary"]["symbols_at_target_fixed"] * runs["frozen"][x]["summary"]["tokens_per_message"] for x in common]))
    cases = (
        ("frozen_fixed", "frozen receiver, fixed rate", ORANGE, "o", (-15, -11), (0, (4, 2))),
        ("frozen", "frozen receiver + refinement", ORANGE, "s", (-18, -11), "-"),
        ("context_only", "context memory + refinement", AQUA, "^", (17, 7), "-"),
    )
    for ref, label, color, marker, offset, style in cases:
        change = 100 * ((payload["nested"] + bits["nested"] / (cap * n)) / payload[ref] - 1)
        ax.plot(n, change, color=color, lw=1.7, ls=style, label=f"vs {label}")
        star = bits["nested"] / (cap * (payload[ref] - payload["nested"]))
        ax.plot([star], [0], marker=marker, color=color, ms=6, mec="white", mew=0.8, zorder=3)
        ax.annotate(f"{star:,.0f}", (star, 0), xytext=offset, textcoords="offset points", ha="center", fontsize=7, color=INK)
        print(f"[netrate] seeds {common}; DN-SC vs {ref}: N* = {star:.0f}, asymptote {100 * (payload['nested'] / payload[ref] - 1):+.1f} %")
    ax.axhline(0, color=MUTED, lw=0.9)
    ax.set_xscale("log")
    ax.set_xlim(30, 1e5)
    ax.set_ylim(-25, 20)
    ax.set_xlabel("Messages served per adaptation step, $N$")
    ax.set_ylabel("Change in net semantic rate (%)")
    tidy(ax)
    ax.legend(loc="upper right", fontsize=7.5, borderaxespad=0.6)
    fig.tight_layout()
    save(fig, out, "fig_netrate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="paper/figures")
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    style()
    runs = runs_by_method()
    fig_operating(out, runs)
    fig_prior_stream(out, runs)
    fig_netrate(out, runs)


if __name__ == "__main__":
    main()
