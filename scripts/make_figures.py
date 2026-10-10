"""Draw the figures from the raw result files.

    python scripts/make_figures.py --config configs/main.yaml

Axes are deliberately not zoomed and seed variability is drawn as a band, so
small differences are not magnified into apparent structure.
"""

from __future__ import annotations

import glob
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dnsc.cli import make_parser, parse  # noqa: E402
from dnsc.utils import ensure_dir, load_json  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150, "savefig.bbox": "tight"})


def smooth(values, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def rate_figures(out_dir: Path, figs: Path) -> None:
    styles = {"llm": ("-", "o"), "unigram": ("--", "s"), "none": (":", "^")}
    for path in sorted(glob.glob(str(out_dir / "eval" / "rate_*.json"))):
        data = load_json(path)
        frame = pd.DataFrame(data["rows"])
        channels = list(dict.fromkeys(frame["channel"]))
        fig, axes = plt.subplots(1, len(channels), figsize=(4.4 * len(channels), 3.4), squeeze=False)
        for ax, channel in zip(axes[0], channels):
            sub = frame[frame["channel"] == channel]
            colour = {snr: f"C{i}" for i, snr in enumerate(sorted(sub["snr_db"].unique()))}  # one colour per SNR
            for prior, (line, marker) in styles.items():
                for snr, group in sub[sub["system"] == f"dnsc_{prior}"].groupby("snr_db"):
                    ax.plot(group["m"], group["bleu4"], linestyle=line, marker=marker, markersize=3, color=colour[snr], label=f"{prior} prior, {snr:g} dB")
            ax.set_xscale("log", base=2)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel("real channel symbols per token, m")
            ax.set_ylabel("BLEU-4")
            ax.set_title(f"{data['corpus']} - {channel}")
        axes[0][-1].legend(fontsize=5, ncol=3, loc="lower right")
        fig.savefig(figs / f"rate_{data['corpus']}.pdf")
        fig.savefig(figs / f"rate_{data['corpus']}.png")
        plt.close(fig)


def stream_figures(out_dir: Path, figs: Path) -> None:
    by_kind = defaultdict(lambda: defaultdict(list))
    for path in sorted(glob.glob(str(out_dir / "stream" / "*.json"))):
        result = load_json(path)
        by_kind[result["stream"]][result["method"]].append(result)
    for kind, methods in by_kind.items():
        any_run = next(iter(methods.values()))[0]
        n_steps = any_run["n_steps"]
        window = max(5, n_steps // 60)
        fig, (ax_acc, ax_bits) = plt.subplots(2, 1, figsize=(7.0, 5.0), sharex=False)
        for method, runs in sorted(methods.items()):
            curves = np.stack([smooth(r["log"]["tok_acc"], window) for r in runs])
            x = np.arange(curves.shape[1]) + window
            mean, sd = curves.mean(0), curves.std(0)
            (line,) = ax_acc.plot(x, mean, label=f"{method} (n={len(runs)})", linewidth=1.2)
            ax_acc.fill_between(x, mean - sd, mean + sd, color=line.get_color(), alpha=0.15)
            cumulative = np.mean([r["ledger"]["cumulative"] for r in runs], axis=0) / 1e6
            ax_bits.plot(np.arange(1, n_steps + 1), cumulative, label=method, color=line.get_color(), linewidth=1.2)
        regime = np.asarray(any_run["log"]["regime"])
        for boundary in np.where(np.diff(regime) != 0)[0] + 1:
            ax_acc.axvline(boundary, color="k", linewidth=0.5, alpha=0.4)
            ax_bits.axvline(boundary, color="k", linewidth=0.5, alpha=0.4)
        ax_acc.set_ylabel(f"prequential token accuracy\n(moving average, {window} steps)")
        ax_acc.set_ylim(0, 1.0)
        ax_acc.legend(fontsize=6, ncol=3)
        ax_acc.set_title(f"'{kind}' drift stream (vertical lines: slow-regime switches)")
        ax_bits.set_ylabel("cumulative sync (Mbit)")
        ax_bits.set_xlabel("stream step")
        fig.savefig(figs / f"stream_{kind}.pdf")
        fig.savefig(figs / f"stream_{kind}.png")
        plt.close(fig)


def refine_figures(out_dir: Path, figs: Path) -> None:
    for path in sorted(glob.glob(str(out_dir / "eval" / "refine_*.json"))):
        data = load_json(path)
        results = data["results"]
        channels = list(dict.fromkeys(r["channel"] for r in results))
        fig, axes = plt.subplots(1, len(channels), figsize=(4.4 * len(channels), 3.4), squeeze=False)
        for ax, channel in zip(axes[0], channels):
            for res in (r for r in results if r["channel"] == channel):
                curve = sorted(res["curve"], key=lambda c: c["symbols"])
                (line,) = ax.plot([c["symbols"] for c in curve], [c["quality"] for c in curve], label=f"selective, {res['snr_db']:g} dB")
                ax.plot([f["symbols"] for f in res["fixed"]], [f["quality"] for f in res["fixed"]], marker="s", markersize=3, linestyle=":", color=line.get_color(), alpha=0.7)
                ax.scatter([res["oracle"]["symbols"]], [res["oracle"]["quality"]], marker="*", color=line.get_color())
                for point in res.get("free_running", []):
                    ax.scatter([point["symbols"]], [point["gen_tok_acc"]], marker="x", color=line.get_color())
            ax.set_xscale("log", base=2)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel("mean symbols per token")
            ax.set_ylabel("token accuracy")
            ax.set_title(f"{data['corpus']} - {channel}\n(dotted: fixed rate, star: oracle, x: free-running)", fontsize=8)
        axes[0][-1].legend(fontsize=6)
        fig.savefig(figs / f"refine_{data['corpus']}.pdf")
        fig.savefig(figs / f"refine_{data['corpus']}.png")
        plt.close(fig)


def main() -> None:
    cfg, _ = parse(make_parser(__doc__))
    out_dir = Path(cfg["out_dir"])
    figs = ensure_dir(out_dir / "figures")
    rate_figures(out_dir, figs)
    stream_figures(out_dir, figs)
    refine_figures(out_dir, figs)
    print(f"[figures] written to {figs}")


if __name__ == "__main__":
    main()
