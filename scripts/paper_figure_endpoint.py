"""Figure of the primary endpoint for the paper, from the generated tables.

    python scripts/paper_figure_endpoint.py --tables runs/six/tables --out results/gpt2_v0.4_six_seeds/paper

Saving of every receiver against the frozen fixed-rate receiver (same decoder,
same seeds) in real channel symbols per token at 90 % token accuracy, signalling
of refinement charged at its entropy: decoded for real on the audited steps
(with the 95 % paired bootstrap interval of realdecode_hier) and teacher-forced
on every step (refine_cost_hier). Nothing is typed by hand.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REAL, EMULATED, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a63", "#e4e3dd"


def signed(value: float) -> str:
    return f"{value:+.1f}".replace("-", "\u2212")


def percent(cell: str) -> tuple[float, float | None, float | None]:
    numbers = [float(x) for x in re.findall(r"[-+]?\d+\.\d+", cell)]
    return numbers[0], (numbers[1] if len(numbers) > 1 else None), (numbers[2] if len(numbers) > 2 else None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    tables, out = Path(args.tables), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    real = {r["method"]: r for r in csv.DictReader(open(tables / "realdecode_hier.csv", encoding="utf-8"))}
    cost = {r["method"]: r for r in csv.DictReader(open(tables / "refine_cost_hier.csv", encoding="utf-8"))}
    frozen_fixed_tf = float(cost["frozen"]["fixed prefix"])
    # teacher-forced readings against the frozen fixed prefix on the same seeds: the headline table's convention
    seeds_ref = {}
    import json
    import glob

    for path in glob.glob(str(tables.parent / "stream" / "hier_*.json")):
        run = json.load(open(path, encoding="utf-8"))
        seeds_ref.setdefault(run["method"], {})[run["seed"]] = run["summary"]

    def tf_saving(method: str, key: str) -> float:
        seeds = sorted(seeds_ref[method])
        mine = sum(seeds_ref[method][s][key] for s in seeds) / len(seeds)
        ref = sum(seeds_ref["frozen"][s]["symbols_at_target_fixed"] for s in seeds) / len(seeds)
        return 100 * (mine / ref - 1)

    rows = [
        ("Frozen receiver + refinement", real["frozen"]["vs frozen fixed [95% CI]"], tf_saving("frozen", "symbols_at_target_selective_fb_entropy"), int(real["frozen"]["seeds"])),
        ("Context memory + refinement", real["context_only"]["vs frozen fixed [95% CI]"], tf_saving("context_only", "symbols_at_target_selective_fb_entropy"), int(real["context_only"]["seeds"])),
        ("DN-SC memories, fixed rate", real["nested"]["fixed prefix vs frozen fixed"], tf_saving("nested", "symbols_at_target_fixed"), int(real["nested"]["seeds"])),
        ("DN-SC memories + refinement", real["nested"]["vs frozen fixed [95% CI]"], tf_saving("nested", "symbols_at_target_selective_fb_entropy"), int(real["nested"]["seeds"])),
    ]
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK})
    fig, ax = plt.subplots(figsize=(6.6, 2.8), dpi=300)
    ys = list(range(len(rows)))[::-1]
    for y, (label, cell, tf, seeds) in zip(ys, rows):
        value, low, high = percent(cell)
        ax.plot([low, high], [y + 0.16, y + 0.16], color=REAL, lw=2, solid_capstyle="round", zorder=2)
        ax.plot(value, y + 0.16, "o", ms=7, color=REAL, mec="white", mew=1.5, zorder=3)
        ax.text(high + 0.6, y + 0.16, f"{signed(value)} %  [{signed(low)}, {signed(high)}]", va="center", ha="left", color=INK, fontsize=7.5)
        ax.plot(tf, y - 0.16, "D", ms=6, mfc="white", mec=EMULATED, mew=1.8, zorder=3)
        ax.text(tf + 0.9, y - 0.16, f"{signed(tf)} %", va="center", ha="left", color=MUTED, fontsize=7.5)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{label} ({seeds} seeds)" for label, _, _, seeds in rows], fontsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlim(-35, 1)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xticks(range(-35, 1, 5))
    ax.set_xlabel("Change in real channel symbols per token vs. the frozen fixed-rate receiver (%)", fontsize=8.5)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    real_handle = ax.plot([], [], "o-", color=REAL, ms=7, mec="white", mew=1.5, lw=2, label="decoded for real, with 95 % interval (156 audited steps)")[0]
    tf_handle = ax.plot([], [], "D", ms=6, mfc="white", mec=EMULATED, mew=1.8, label="teacher-forced, all 600 steps")[0]
    ax.legend(handles=[real_handle, tf_handle], loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2, frameon=False, fontsize=7.5)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"fig_primary_endpoint.{suffix}", bbox_inches="tight", facecolor="white")
    print(f"[figure] {out / 'fig_primary_endpoint.pdf'}")


if __name__ == "__main__":
    main()
