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
    styles = {"llm_ctx": ("-", "D"), "llm": ("-.", "o"), "unigram": ("--", "s"), "none": (":", "^")}
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
        handles, labels = axes[0][-1].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=6, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.0))
        fig.savefig(figs / f"rate_{data['corpus']}.pdf")
        fig.savefig(figs / f"rate_{data['corpus']}.png")
        plt.close(fig)


ORDER = ["frozen", "context_only", "single_rate", "ewc", "nested_no_context", "nested_no_slots", "nested_two_level", "nested", "full_online"]
STYLE = {"frozen": ("0.45", ":"), "context_only": ("0.15", "--"), "nested": ("C3", "-"), "full_online": ("C0", "-."), "single_rate": ("C1", "-")}


def _ordered(methods: dict) -> list[str]:
    return [m for m in ORDER if m in methods] + sorted(m for m in methods if m not in ORDER)


def _style(method: str, index: int) -> dict:
    colour, line = STYLE.get(method, (f"C{(index % 6) + 4}", "-"))
    return {"color": colour, "linestyle": line, "linewidth": 1.6 if method == "nested" else 1.1}


def stream_figures(cfg, out_dir: Path, figs: Path) -> None:
    by_kind = defaultdict(lambda: defaultdict(dict))
    for path in sorted(glob.glob(str(out_dir / "stream" / "*.json"))):
        result = load_json(path)
        if Path(path).stem.split(f"_{result['method']}_s{result['seed']}", 1)[-1]:
            continue  # a tagged variant run (sensitivity study)
        by_kind[result["stream"]][result["method"]][int(result["seed"])] = result
    for kind, methods in by_kind.items():
        any_run = next(iter(next(iter(methods.values())).values()))
        n_steps = any_run["n_steps"]
        window = max(5, n_steps // 30)
        thresholds = [p["threshold"] for p in any_run["selective_curve"]]
        k_tau = thresholds.index(any_run["threshold"])
        regime = np.asarray(any_run["log"]["regime"])
        boundaries = np.where(np.diff(regime) != 0)[0] + 1

        # ---- 1. the stream over time
        fig, (ax_prior, ax_sym, ax_bits) = plt.subplots(3, 1, figsize=(7.2, 7.0), sharex=True)
        for index, method in enumerate(_ordered(methods)):
            runs = methods[method]
            style = _style(method, index)
            curves = np.stack([smooth(r["log"]["prior_bits"], window) for r in runs.values()])
            x = np.arange(curves.shape[1]) + window
            mean, sd = curves.mean(0), curves.std(0)
            ax_prior.plot(x, mean, label=f"{method} (n={len(runs)})", **style)
            ax_prior.fill_between(x, mean - sd, mean + sd, color=style["color"], alpha=0.12)
            if "frozen" in methods and method != "frozen":  # paired with the frozen receiver on the same messages and noise
                shared = sorted(set(runs) & set(methods["frozen"]))
                if shared:
                    diff = np.stack([smooth(np.asarray(runs[s]["log"]["sel_symbols"])[:, k_tau] - np.asarray(methods["frozen"][s]["log"]["sel_symbols"])[:, k_tau], window) for s in shared])
                    ax_sym.plot(x, diff.mean(0), **style)
                    ax_sym.fill_between(x, diff.mean(0) - diff.std(0), diff.mean(0) + diff.std(0), color=style["color"], alpha=0.12)
            cumulative = np.mean([r["ledger"]["cumulative"] for r in runs.values()], axis=0) / 1e6
            ax_bits.plot(np.arange(1, n_steps + 1), cumulative, **style)
        for ax in (ax_prior, ax_sym, ax_bits):
            for boundary in boundaries:
                ax.axvline(boundary, color="k", linewidth=0.5, alpha=0.4)
        ax_sym.axhline(0.0, color="0.45", linewidth=0.8, linestyle=":")
        ax_prior.set_ylabel(f"prior cross-entropy\n(bits/token, {window}-step mean)")
        ax_sym.set_ylabel(f"symbols/token at confidence {any_run['threshold']:g}\nminus frozen receiver")
        ax_bits.set_yscale("symlog", linthresh=1.0)
        ax_bits.set_ylabel("cumulative sync (Mbit)")
        ax_bits.set_xlabel("stream step")
        ax_prior.legend(fontsize=6, ncol=3)
        ax_prior.set_title(f"'{kind}' drift stream (vertical lines: slow-regime switches; bands: SD over seeds)")
        fig.savefig(figs / f"stream_{kind}.pdf")
        fig.savefig(figs / f"stream_{kind}.png")
        plt.close(fig)

        # ---- 2. operating curves: what fidelity costs in symbols
        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        for index, method in enumerate(_ordered(methods)):
            runs = list(methods[method].values())
            symbols = np.mean([[p["symbols"] for p in r["selective_curve"]] for r in runs], axis=0)
            accuracy = np.mean([[p["tok_acc"] for p in r["selective_curve"]] for r in runs], axis=0)
            ax.plot(symbols, accuracy, marker="o", markersize=3, label=method, **_style(method, index))
        if "frozen" in methods:
            runs = list(methods["frozen"].values())
            ax.plot([p["m"] for p in runs[0]["fixed_curve"]], np.mean([[p["tok_acc"] for p in r["fixed_curve"]] for r in runs], axis=0), marker="s", markersize=3, color="0.7", linestyle="-", label="frozen, fixed prefix")
        ax.axhline(any_run["target_acc"], color="0.45", linewidth=0.8, linestyle=":")
        ax.set_xlabel("mean real symbols per token")
        ax.set_ylabel("token accuracy on the stream")
        ax.set_ylim(0.0, 1.02)
        ax.set_title(f"'{kind}' stream: selective refinement, one point per confidence threshold", fontsize=8)
        ax.legend(fontsize=6)
        fig.savefig(figs / f"curves_{kind}.pdf")
        fig.savefig(figs / f"curves_{kind}.png")
        plt.close(fig)

        # ---- 3. which slot served which part of the stream
        slotted = {m: runs for m, runs in methods.items() if any(r.get("slots", 0) > 1 for r in runs.values())}
        if slotted:
            fig, axes = plt.subplots(len(slotted), 1, figsize=(7.2, 1.3 * len(slotted) + 0.6), sharex=True, squeeze=False)
            for ax, (method, runs) in zip(axes[:, 0], sorted(slotted.items())):
                for row, (seed, run) in enumerate(sorted(runs.items())):
                    ax.scatter(np.arange(1, n_steps + 1), np.full(n_steps, row), c=np.asarray(run["log"]["slot"]), cmap="viridis", vmin=-1, vmax=max(run["slots"] - 1, 1), marker="|", s=40)
                for boundary in boundaries:
                    ax.axvline(boundary, color="k", linewidth=0.5, alpha=0.4)
                ax.set_yticks(range(len(runs)))
                ax.set_yticklabels([f"seed {seed}" for seed in sorted(runs)], fontsize=6)
                ax.set_title(f"{method}: active LoRA slot (darkest: none, the frozen prior)", fontsize=7)
                ax.grid(False)
            axes[-1, 0].set_xlabel("stream step")
            fig.savefig(figs / f"slots_{kind}.pdf")
            fig.savefig(figs / f"slots_{kind}.png")
            plt.close(fig)


def refine_cost_figures(out_dir: Path, figs: Path) -> None:
    """What selective refinement costs: for the receiver without memory and
    for the proposed one, the fixed-prefix curve next to the selective curve
    forward only, with its return-link signalling (entropy and bitmap), and
    the free-running audit. Series differ by marker and line style as well as
    by colour, so the figure also reads in greyscale."""
    by_kind = defaultdict(lambda: defaultdict(dict))
    for path in sorted(glob.glob(str(out_dir / "stream" / "*.json"))):
        result = load_json(path)
        if Path(path).stem.split(f"_{result['method']}_s{result['seed']}", 1)[-1]:
            continue
        by_kind[result["stream"]][result["method"]][int(result["seed"])] = result
    fixed_colour, selective_colour = "#d95f02", "#1f77b4"  # checked for colour-vision deficiencies; identity is also in the markers
    for kind, methods in by_kind.items():
        shown = [m for m in ("frozen", "nested") if m in methods and all("symbols_total" in r["selective_curve"][0] for r in methods[m].values())]
        if not shown:
            continue  # runs of a version that did not record the signalling
        fig, axes = plt.subplots(1, len(shown), figsize=(4.6 * len(shown), 3.6), squeeze=False, sharey=True)
        for ax, method in zip(axes[0], shown):
            runs = list(methods[method].values())
            mean = lambda pick: np.mean([[pick(p) for p in r["selective_curve"]] for r in runs], axis=0)  # noqa: E731
            accuracy = mean(lambda p: p["tok_acc"])
            ax.plot([p["m"] for p in runs[0]["fixed_curve"]], np.mean([[p["tok_acc"] for p in r["fixed_curve"]] for r in runs], axis=0), color=fixed_colour, linestyle="-", marker="s", markersize=4, linewidth=1.2, label="fixed prefix (no signalling)")
            ax.plot(mean(lambda p: p["symbols"]), accuracy, color=selective_colour, linestyle="--", marker="o", markersize=4, linewidth=1.2, markerfacecolor="white", label="selective, forward symbols only")
            ax.plot(mean(lambda p: p["symbols_total_entropy"]), accuracy, color=selective_colour, linestyle="-", marker="o", markersize=4, linewidth=1.8, label="selective + signalling (entropy)")
            ax.plot(mean(lambda p: p["symbols_total"]), accuracy, color=selective_colour, linestyle=":", marker="^", markersize=4, linewidth=1.2, label="selective + signalling (bitmap)")
            live = [r["live_curve"] for r in runs if r.get("live_curve")]
            if live:
                ax.scatter(np.mean([[p["symbols"] for p in curve] for curve in live], axis=0), np.mean([[p["tok_acc"] for p in curve] for curve in live], axis=0), marker="x", s=36, color="black", linewidths=1.2, zorder=3, label="selective decoded for real, forward only (audited steps)")
            live_fixed = [r["live_fixed_curve"] for r in runs if r.get("live_fixed_curve")]
            if live_fixed:
                ax.scatter([p["m"] for p in live_fixed[0]], np.mean([[p["tok_acc"] for p in curve] for curve in live_fixed], axis=0), marker="+", s=48, color="black", linewidths=1.2, zorder=3, label="fixed prefix decoded for real (audited steps)")
            ax.axhline(runs[0]["target_acc"], color="0.45", linewidth=0.8, linestyle=":")
            ax.set_xlabel("real symbols per token")
            ax.set_ylim(0.0, 1.02)
            ax.set_title(f"{method} (n={len(runs)})", fontsize=8)
        axes[0][0].set_ylabel("token accuracy on the stream")
        axes[0][-1].legend(fontsize=6, loc="lower right")
        fig.suptitle(f"'{kind}' stream: what selective refinement costs once its return-link signalling is charged", fontsize=8)
        fig.savefig(figs / f"refine_cost_{kind}.pdf")
        fig.savefig(figs / f"refine_cost_{kind}.png")
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
    stream_figures(cfg, out_dir, figs)
    refine_cost_figures(out_dir, figs)
    refine_figures(out_dir, figs)
    print(f"[figures] written to {figs}")


if __name__ == "__main__":
    main()
