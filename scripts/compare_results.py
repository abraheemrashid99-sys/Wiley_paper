"""Compare a reproduction with the published results, number by number.

    python scripts/compare_results.py --reference results/gpt2 --new results_repro/gpt2

Both folders are exports written by ``scripts/export_results.py``. The script
writes ``<new>/COMPARISON.md`` and prints it. It reports

* the headline numbers of every stream and method, side by side;
* whether each conclusion of the paper holds in the reproduction;
* every stream run, every evaluation file and every table, with the largest
  difference found.

Two runs on different hardware are not bit-identical (floating-point
arithmetic differs between a CPU and a GPU, and the online adaptation
amplifies such differences a little), so agreement is judged against a
tolerance and against the conclusions. The script never fails on a missing
file: whatever cannot be compared is listed as such.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

HEADLINE = [
    ("mean_prior_bits", "prior bits/token"),
    ("symbols_at_target_selective", "symbols/token at target (selective)"),
    ("symbols_at_target_fixed", "symbols/token at target (fixed)"),
    ("mean_tok_acc", "token accuracy at m_op"),
    ("sync_bits_per_step", "sync bits/step"),
]
SKIP_KEYS = {"seconds", "env", "config", "time", "calibrator", "it_per_s"}
NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def load(path: Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def fmt(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def rel(new, ref) -> float:
    if new is None or ref is None:
        return float("nan")
    return abs(new - ref) / max(abs(ref), 1e-9)


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def stream_runs(folder: Path) -> dict[str, dict]:
    out = {}
    for path in sorted((folder / "stream").glob("*.json")):
        try:
            out[path.stem] = load(path)
        except Exception as error:  # an unreadable file is reported, not fatal
            print(f"[compare] cannot read {path}: {error}")
    return out


def tag_of(stem: str, run: dict) -> str:
    return stem.split(f"_{run['method']}_s{run['seed']}", 1)[-1].lstrip("_")


def by_method(runs: dict[str, dict], names: set[str]) -> dict[tuple[str, str], dict[str, list]]:
    """Untagged runs grouped by (stream, method), restricted to ``names``."""
    groups: dict[tuple[str, str], dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for stem, run in runs.items():
        if stem not in names or tag_of(stem, run):
            continue
        for key, _ in HEADLINE:
            groups[(run["stream"], run["method"])][key].append(run["summary"].get(key))
    return groups


def numeric_leaves(obj, prefix: str = "") -> dict[str, float]:
    out = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key not in SKIP_KEYS:
                out.update(numeric_leaves(value, f"{prefix}/{key}"))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            label = index
            if isinstance(value, dict):  # name rows by their identifying fields when they have any
                ident = [str(value[k]) for k in ("system", "prior", "domain", "variant", "measured_on", "channel", "snr_db", "m", "threshold", "step") if k in value]
                label = ",".join(ident) if ident else index
            out.update(numeric_leaves(value, f"{prefix}[{label}]"))
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        out[prefix] = float(obj)
    return out


def diff_leaves(ref, new) -> tuple[int, float, str, float]:
    """(number of shared numbers, largest absolute difference, where, median absolute difference)."""
    a, b = numeric_leaves(ref), numeric_leaves(new)
    shared = [key for key in a if key in b and not (math.isnan(a[key]) or math.isnan(b[key]))]
    if not shared:
        return 0, float("nan"), "-", float("nan")
    diffs = sorted((abs(a[key] - b[key]), key) for key in shared)
    return len(shared), diffs[-1][0], diffs[-1][1], diffs[len(diffs) // 2][0]


def table_numbers(path: Path) -> list[float]:
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    return [float(match) for row in rows[1:] for cell in row for match in NUMBER.findall(cell.replace("−", "-"))]


def conclusions(runs: dict[str, dict], folder: Path) -> list[tuple[str, str, bool | None]]:
    """(statement, measured value, holds) for the claims of the paper,
    computed from the untagged runs of ``runs``."""
    groups = by_method(runs, set(runs))

    def value(stream, method, key):
        return mean(groups.get((stream, method), {}).get(key, []))

    out = []

    def add(text, shown, holds):
        out.append((text, shown, holds))

    def check(text, compute):
        try:
            shown, holds = compute()
        except Exception:
            shown, holds = "not available", None
        add(text, shown, holds)

    def refinement():
        fixed, sel = value("hier", "frozen", "symbols_at_target_fixed"), value("hier", "frozen", "symbols_at_target_selective")
        return f"{sel / fixed - 1:+.1%}", sel < 0.8 * fixed

    def nested_total():
        fixed, sel = value("hier", "frozen", "symbols_at_target_fixed"), value("hier", "nested", "symbols_at_target_selective")
        return f"{sel / fixed - 1:+.1%}", sel < 0.6 * fixed

    def ordering(stream, methods, key="mean_prior_bits"):
        def compute():
            values = [value(stream, m, key) for m in methods]
            return " > ".join(fmt(v) for v in values), all(a > b for a, b in zip(values, values[1:]))

        return compute

    def slots():
        a, b = value("hier", "nested_no_slots", "mean_prior_bits"), value("hier", "nested", "mean_prior_bits")
        return f"{a - b:+.3f} bits", a > b

    def single():
        a, b = value("hier", "single_rate", "mean_prior_bits"), value("hier", "nested", "mean_prior_bits")
        return f"{a - b:+.3f} bits", a > b

    def upper():
        a, b = value("hier", "nested", "mean_prior_bits"), value("hier", "full_online", "mean_prior_bits")
        return f"{a - b:+.3f} bits", abs(a - b) < 0.1

    def sync_ratio():
        a, b = value("hier", "full_online", "sync_bits_per_step"), value("hier", "nested", "sync_bits_per_step")
        return f"{a / b:.0f}x", a > 20 * b

    def temporal_tie():
        a, b = value("temporal", "single_rate", "mean_prior_bits"), value("temporal", "nested", "mean_prior_bits")
        return f"{a - b:+.3f} bits", abs(a - b) < 0.1

    def codebook():
        nested = load(folder / "base" / "pretrain_summary.json")["history"][-1]
        fixed_dir = sorted(folder.glob("fixed*"))[0]
        m = fixed_dir.name.replace("fixed", "")
        fixed = load(fixed_dir / "pretrain_summary.json")["history"][-1]
        return f"{nested[m]:.4f} nested vs {fixed[m]:.4f} fixed-rate", abs(nested[m] - fixed[m]) < 0.02

    def rate_order():
        rows = load(sorted((folder / "eval").glob("rate_*.json"))[0])["rows"]
        grans = sorted({r["m"] for r in rows})
        m = min(grans, key=lambda g: abs(g - 6))
        values = []
        for prior in ("llm_ctx", "llm", "unigram", "none"):
            pick = [r["bleu4"] for r in rows if r.get("prior") == prior and r["channel"] == "awgn" and abs(r["snr_db"] - 10.0) < 1e-6 and r["m"] == m]
            values.append(pick[0])
        return " > ".join(fmt(v) for v in values), all(a > b for a, b in zip(values, values[1:]))

    check("Selective refinement alone saves at least 20 % of the symbols of a fixed-rate receiver (hierarchical stream, frozen prior)", refinement)
    check("The full system (nested) saves at least 40 % of the symbols of a fixed-rate frozen receiver (hierarchical stream)", nested_total)
    check("Prior bits fall in the order frozen > context_only > single_rate > nested (hierarchical stream)", ordering("hier", ["frozen", "context_only", "single_rate", "nested"]))
    check("Symbols at the target accuracy fall in the order frozen > context_only > single_rate > nested (hierarchical stream)", ordering("hier", ["frozen", "context_only", "single_rate", "nested"], "symbols_at_target_selective"))
    check("Nested beats the single-rate baseline at a matched synchronisation budget (hierarchical stream; single_rate − nested > 0)", single)
    check("The slots are what helps: nested_no_slots − nested > 0 (hierarchical stream)", slots)
    check("Nested is within 0.1 bit of the full-synchronisation upper reference (hierarchical stream)", upper)
    check("The upper reference needs more than 20 times the synchronisation traffic of nested", sync_ratio)
    check("Prior bits fall in the order frozen > context_only > nested (temporal stream)", ordering("temporal", ["frozen", "context_only", "nested"]))
    check("On the temporal stream nested and single-rate are tied (within 0.1 bit)", temporal_tie)
    check("The nested codebook loses nothing against a fixed-rate codebook (accuracy difference below 0.02)", codebook)
    check("BLEU-4 at AWGN 10 dB improves with every part of the prior: llm_ctx > llm > unigram > none", rate_order)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--tolerance", type=float, default=0.03, help="relative tolerance on the headline numbers (default 3 %%)")
    args = parser.parse_args()
    ref_dir, new_dir = Path(args.reference), Path(args.new)
    lines: list[str] = ["# Reproduction check", "", f"Reference: `{ref_dir}`  ·  reproduction: `{new_dir}`  ·  tolerance on headline numbers: {args.tolerance:.0%}", ""]
    problems: list[str] = []

    ref_runs, new_runs = stream_runs(ref_dir), stream_runs(new_dir)
    shared = sorted(set(ref_runs) & set(new_runs))
    missing = sorted(set(ref_runs) - set(new_runs))

    # --- environments
    try:
        a, b = next(iter(ref_runs.values())), next(iter(new_runs.values()))
        lines += ["## Environments", "", "| | device | torch | transformers | python | code |", "| --- | --- | --- | --- | --- | --- |"]
        for label, run in (("reference", a), ("reproduction", b)):
            env = run.get("env", {})
            device = str(run.get("config", {}).get("device", "?")) + (f" ({env['gpu']})" if env.get("gpu") else "")
            lines.append(f"| {label} | {device} | {env.get('torch', '?')} | {env.get('transformers', '?')} | {env.get('python', '?')} | {env.get('git', '?')} |")
        lines.append("")
    except Exception:
        pass

    # --- headline per stream and method
    lines += ["## Headline numbers (mean over the seeds present in both folders)", ""]
    ref_groups, new_groups = by_method(ref_runs, set(shared)), by_method(new_runs, set(shared))
    worst = 0.0
    if ref_groups:
        lines += ["| Stream | Method | Seeds | Prior bits/token: published | reproduced | diff | Symbols/token at target: published | reproduced | diff | Within tolerance |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for (stream, method), group in sorted(ref_groups.items()):
            other = new_groups.get((stream, method), {})
            p_ref, p_new = mean(group["mean_prior_bits"]), mean(other.get("mean_prior_bits", []))
            s_ref, s_new = mean(group["symbols_at_target_selective"]), mean(other.get("symbols_at_target_selective", []))
            gaps = [rel(p_new, p_ref)] + ([rel(s_new, s_ref)] if s_ref is not None and s_new is not None else [])
            gap = max(g for g in gaps if not math.isnan(g)) if any(not math.isnan(g) for g in gaps) else float("nan")
            both_or_neither = (s_ref is None) == (s_new is None)
            ok = (not math.isnan(gap)) and gap <= args.tolerance and both_or_neither
            worst = max(worst, 0.0 if math.isnan(gap) else gap)
            if not ok:
                problems.append(f"{stream}/{method}: headline numbers differ by {gap:.1%}" if both_or_neither else f"{stream}/{method}: the target accuracy is reached in only one of the two")
            d_p = "n/a" if p_new is None else f"{p_new - p_ref:+.3f}"
            d_s = "n/a" if s_ref is None or s_new is None else f"{s_new - s_ref:+.3f}"
            lines.append(f"| {stream} | {method} | {len(group['mean_prior_bits'])} | {fmt(p_ref)} | {fmt(p_new)} | {d_p} | {fmt(s_ref)} | {fmt(s_new)} | {d_s} | {'yes' if ok else 'NO'} |")
        lines += ["", f"Largest relative difference of a headline number: {worst:.2%}.", ""]
    else:
        lines += ["No stream runs are present in both folders.", ""]
        problems.append("no stream runs to compare")
    if missing:
        problems.append(f"{len(missing)} stream runs of the reference are missing in the reproduction")
        lines += [f"Missing in the reproduction ({len(missing)}): " + ", ".join(f"`{m}`" for m in missing), ""]

    # --- conclusions
    lines += ["## Conclusions of the paper", "", "| Conclusion | Published | Reproduced | Holds in the reproduction |", "| --- | --- | --- | --- |"]
    try:
        for (text, shown_ref, holds_ref), (_, shown_new, holds_new) in zip(conclusions(ref_runs, ref_dir), conclusions(new_runs, new_dir)):
            verdict = "n/a" if holds_new is None else ("yes" if holds_new else "NO")
            if holds_new is False and holds_ref:
                problems.append(f"conclusion not reproduced: {text}")
            if holds_new is None and holds_ref is not None:
                problems.append(f"conclusion could not be checked: {text}")
            lines.append(f"| {text} | {shown_ref} | {shown_new} | {verdict} |")
    except Exception as error:
        lines.append(f"| (could not be evaluated: {error}) | | | |")
        problems.append("the conclusions could not be evaluated")
    lines.append("")

    # --- every stream run
    lines += ["## Every stream run", "", "| Run | Prior bits: published | reproduced | Symbols at target: published | reproduced | Token accuracy: published | reproduced | Sync kbit/step: published | reproduced |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for stem in shared:
        a, b = ref_runs[stem]["summary"], new_runs[stem]["summary"]
        lines.append(
            f"| {stem} | {fmt(a.get('mean_prior_bits'))} | {fmt(b.get('mean_prior_bits'))} | {fmt(a.get('symbols_at_target_selective'))} | {fmt(b.get('symbols_at_target_selective'))} "
            f"| {fmt(a.get('mean_tok_acc'), 4)} | {fmt(b.get('mean_tok_acc'), 4)} | {fmt((a.get('sync_bits_per_step') or 0) / 1e3, 2)} | {fmt((b.get('sync_bits_per_step') or 0) / 1e3, 2)} |"
        )
    lines.append("")

    # --- evaluation files and codebooks
    lines += ["## Evaluation files and codebooks (all numbers they contain)", "", "| File | Numbers compared | Median abs. difference | Largest abs. difference | Where |", "| --- | --- | --- | --- | --- |"]
    files = sorted({p.relative_to(ref_dir) for pattern in ("eval/*.json", "base/pretrain_summary.json", "fixed*/pretrain_summary.json") for p in ref_dir.glob(pattern)})
    for name in files:
        if not (new_dir / name).exists():
            lines.append(f"| {name} | missing in the reproduction | | | |")
            problems.append(f"{name} is missing in the reproduction")
            continue
        try:
            count, largest, where, median = diff_leaves(load(ref_dir / name), load(new_dir / name))
            lines.append(f"| {name} | {count} | {fmt(median, 4)} | {fmt(largest, 4)} | `{where}` |")
        except Exception as error:
            lines.append(f"| {name} | could not be compared ({error}) | | | |")
    lines.append("")

    # --- tables
    lines += ["## Generated tables (every number in every table)", "", "| Table | Numbers | Identical to 3 decimals | Median abs. difference | Largest abs. difference |", "| --- | --- | --- | --- | --- |"]
    for path in sorted((ref_dir / "tables").glob("*.csv")):
        other = new_dir / "tables" / path.name
        if not other.exists():
            lines.append(f"| {path.stem} | missing in the reproduction | | | |")
            continue
        try:
            a, b = table_numbers(path), table_numbers(other)
            if len(a) != len(b):
                lines.append(f"| {path.stem} | {len(a)} vs {len(b)} (different layout, e.g. a target reached in only one of them) | | | |")
                continue
            diffs = sorted(abs(x - y) for x, y in zip(a, b))
            same = sum(d < 5e-4 for d in diffs)
            lines.append(f"| {path.stem} | {len(a)} | {same} | {fmt(diffs[len(diffs) // 2], 4) if diffs else 'n/a'} | {fmt(diffs[-1], 4) if diffs else 'n/a'} |")
        except Exception as error:
            lines.append(f"| {path.stem} | could not be compared ({error}) | | | |")
    lines.append("")

    verdict = "REPRODUCED: every headline number is within the tolerance and every conclusion holds." if not problems else "CHECK NEEDED:"
    head = ["## Verdict", "", f"**{verdict}**", ""] + [f"- {p}" for p in problems] + ([""] if problems else [])
    lines = lines[:4] + head + lines[4:]
    text = "\n".join(lines)
    try:
        new_dir.mkdir(parents=True, exist_ok=True)
        (new_dir / "COMPARISON.md").write_text(text, encoding="utf-8")
    except Exception as error:
        print(f"[compare] could not write COMPARISON.md: {error}")
    print(text)
    print(f"\n[compare] wrote {new_dir / 'COMPARISON.md'}")


if __name__ == "__main__":
    main()
