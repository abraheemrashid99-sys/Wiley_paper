"""Assemble the exported tables and figures into one readable report.

    python scripts/write_report.py --config configs/gpt2.yaml --to results/gpt2

Writes ``<to>/TABLES.md``: the headline comparison computed from the stream
results, followed by every generated table and a link to every figure. Nothing
in it is typed by hand; the interpretation lives in ``<to>/RESULTS.md``.
"""

from __future__ import annotations

import glob
from collections import defaultdict
from pathlib import Path

import numpy as np

from dnsc.cli import make_parser, parse
from dnsc.linkbudget import pooled_live
from dnsc.utils import load_json

ORDER = ["frozen", "context_only", "single_rate", "ewc", "nested_no_context", "nested_no_slots", "nested_two_level", "nested", "full_online"]
SECTIONS = [
    ("Rate and fidelity on the base corpus", ["rate_*_bleu4", "rate_*_sim", "rate_*_chrf", "rate_*_wer", "rate_*_gen_tok_acc", "rate_*_tok_acc", "rate_*_separation", "rate_*_power"]),
    ("Decoder of selective refinement (validation text)", ["decoder_selection"]),
    ("Selective refinement on the base corpus", ["refine_*"]),
    ("What each memory could buy (diagnostic with oracle inputs)", ["headroom"]),
    ("Hierarchical drift stream", ["stream_hier", "realdecode_hier", "emulation_check_hier", "refine_cost_hier", "refine_start_hier", "targets_hier", "equivalence_hier", "regimes_hier", "curves_hier", "netrate_hier"]),
    ("Temporal drift stream", ["stream_temporal", "realdecode_temporal", "emulation_check_temporal", "refine_cost_temporal", "refine_start_temporal", "targets_temporal", "equivalence_temporal", "regimes_temporal", "curves_temporal", "netrate_temporal"]),
    ("Variant runs", ["sensitivity"]),
    ("Measured computational cost", ["cost"]),
]


STREAM_REFINE = ("refine_cost_", "refine_start_")


def ordinal(n) -> str:
    if not isinstance(n, int):
        return "?th"
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def headline(stream_dir: Path, kind: str) -> list[str]:
    runs = defaultdict(dict)
    for path in sorted(glob.glob(str(stream_dir / f"{kind}_*.json"))):
        result = load_json(path)
        if Path(path).stem.split(f"_{result['method']}_s{result['seed']}", 1)[-1]:
            continue  # tagged variant run
        runs[result["method"]][int(result["seed"])] = result
    if "frozen" not in runs:
        return []

    def need(method: str, key: str, seeds: list[int] | None = None) -> tuple[float, int]:
        chosen = runs[method] if seeds is None else {s: runs[method][s] for s in seeds if s in runs[method]}
        values = [r["summary"].get(key) for r in chosen.values()]
        if seeds is not None and len(values) < len(seeds):
            return float("nan"), len(values)  # the reference lacks some of these seeds
        return (float("nan") if any(v is None for v in values) else float(np.mean(values))), len(values)

    def show(value: float, pattern: str = "{:.3f}") -> str:
        return "n/a" if np.isnan(value) else pattern.format(value)

    target = next(iter(runs["frozen"].values()))["target_acc"]
    fixed, n_fixed = need("frozen", "symbols_at_target_fixed")
    lines = [
        f"Real channel symbols per token needed for {target:.0%} token accuracy on the '{kind}' stream, teacher-forced on every step (mean over seeds; read off each method's measured operating curve). Every receiver is compared with the frozen receiver on the same seeds.",
        "",
        "| Receiver | Seeds | Prior bits/token | Symbols/token | vs. frozen at a fixed rate | vs. frozen with refinement | Sync kbit/step | Symbols/token incl. feedback (entropy) | vs. frozen at a fixed rate, feedback charged |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        f"| frozen, fixed prefix length | {n_fixed} | {need('frozen', 'mean_prior_bits')[0]:.3f} | {show(fixed)} | — | — | 0 | {show(fixed)} | — |",
    ]
    for method in [m for m in ORDER if m in runs] + sorted(m for m in runs if m not in ORDER):
        seeds = sorted(runs[method])
        value, n = need(method, "symbols_at_target_selective")
        prior, _ = need(method, "mean_prior_bits")
        sync, _ = need(method, "sync_bits_per_step")
        ref_fixed, _ = need("frozen", "symbols_at_target_fixed", seeds)
        ref_sel, _ = need("frozen", "symbols_at_target_selective", seeds)
        label = "frozen, selective refinement" if method == "frozen" else f"{method}, selective refinement"
        versus = "—" if method == "frozen" else show(value / ref_sel - 1, "{:+.1%}")
        total, _ = need(method, "symbols_at_target_selective_fb_entropy")
        lines.append(f"| {label} | {n} | {prior:.3f} | {show(value)} | {show(value / ref_fixed - 1, '{:+.1%}')} | {versus} | {sync / 1e3:.2f} | {show(total)} | {show(total / ref_fixed - 1, '{:+.1%}')} |")
    lines += ["", "\"Incl. feedback\": forward symbols plus the return-link signalling of the refinement requests, charged at its entropy and at the capacity of the mean SNR (derived; see the refinement-cost table for the bitmap variant and the free-running audit). n/a: not recorded by the code version that produced the run, or the target is not reached."]

    # the same comparison decoded for real (free-running audit), both sides with the same decoder;
    # every receiver is compared with the frozen receiver on the same seeds (the same audited messages)
    if "frozen" in runs and all(r.get("live_fixed_curve") for r in runs["frozen"].values()):
        first = next(iter(runs["frozen"].values()))
        decoder = first["summary"].get("decoder", {"beam": 1, "start_m": None, "reopen": False})
        shell = "the smallest" if decoder["start_m"] is None else f"{decoder['start_m']} symbols"
        lines += [
            "",
            f"**Decoded for real.** The same symbols decoded free-running on the audited steps (every {ordinal(first.get('live_every'))} step of every run, pooled over the seeds of a method) with the decoder of the run (beam {decoder['beam']}, first shell {shell}, reopening {'on' if decoder['reopen'] else 'off'}), for selective refinement and the fixed-prefix reference alike; selective refinement is charged its signalling. Every receiver is compared with the frozen receiver's fixed prefix on the same seeds.",
            "",
            "| Receiver | Seeds | Audited steps | Fixed prefix | vs. frozen fixed prefix | Selective + signalling (entropy) | vs. frozen fixed prefix | Selective + signalling (bitmap) | vs. frozen fixed prefix |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        num = lambda v: "not reached" if v is None else f"{v:.3f}"  # noqa: E731
        for method in [m for m in ORDER if m in runs] + sorted(m for m in runs if m not in ORDER):
            seeds = sorted(runs[method])
            mine = pooled_live([runs[method][seed] for seed in seeds], target)
            frozen = [runs["frozen"][seed] for seed in seeds if seed in runs["frozen"]]
            reference = pooled_live(frozen, target) if len(frozen) == len(seeds) else None
            if mine is None:
                continue
            ref = None if reference is None else reference["fixed"]
            rel = lambda v: "n/a" if v is None or ref is None else f"{v / ref - 1:+.1%}"  # noqa: E731,B023
            lines.append(f"| {method} | {len(seeds)} | {mine['audited_steps']} | {num(mine['fixed'])} | {'—' if method == 'frozen' else rel(mine['fixed'])} | {num(mine['total_entropy'])} | {rel(mine['total_entropy'])} | {num(mine['total_bitmap'])} | {rel(mine['total_bitmap'])} |")
    return lines + [""]


def main() -> None:
    parser = make_parser(__doc__)
    parser.add_argument("--to", required=True, help="exported results folder (see export_results.py)")
    cfg, args = parse(parser)
    out, src = Path(args.to), Path(cfg["out_dir"])
    tables, figures = out / "tables", out / "figures"
    lines = [f"# Tables and figures: {cfg['name']}", "", "Generated by `scripts/write_report.py` from the result files; nothing here is typed by hand.", ""]
    for kind in ("hier", "temporal"):
        block = headline(src / "stream", kind)
        if block:
            lines += [f"## Headline, {kind} stream", ""] + block
    used = set()
    for title, patterns in SECTIONS:
        files = []
        for pattern in patterns:  # a table is printed once, under the first pattern that matches it
            for path in sorted(tables.glob(pattern + ".md")):
                if pattern == "refine_*" and path.stem.startswith(STREAM_REFINE):
                    continue  # stream tables with a "refine_" name belong to the stream sections below
                if path not in used:
                    used.add(path)
                    files.append(path)
        if not files:
            continue
        lines += [f"## {title}", ""]
        for path in files:
            lines += [path.read_text(encoding="utf-8").strip(), ""]
    rest = [f for f in sorted(tables.glob("*.md")) if f not in used]
    if rest:
        lines += ["## Other tables", ""]
        for path in rest:
            lines += [path.read_text(encoding="utf-8").strip(), ""]
    pictures = sorted(figures.glob("*.png"))
    if pictures:
        lines += ["## Figures", ""]
        for path in pictures:
            lines += [f"### {path.stem}", "", f"![{path.stem}](figures/{path.name})", ""]
    (out / "TABLES.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] wrote {out / 'TABLES.md'}")


if __name__ == "__main__":
    main()
