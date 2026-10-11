"""Re-run the whole published study from scratch and compare with the published numbers.

    python scripts/reproduce.py --config configs/gpt2.yaml --out runs/gpt2_repro \
        --export results_repro/gpt2 --reference results/gpt2 device=auto

Runs, in order, exactly the experiments behind ``results/gpt2``: data,
codebook, the hierarchical stream (main methods on three seeds, ablations),
the temporal stream, the variant runs, the diagnostics and evaluations on the
base corpus, the fixed-rate codebook check; then the tables, figures and the
report, and finally a comparison with the reference folder
(``COMPARISON.md`` in the export folder).

Everything that is already finished is skipped, so the command can simply be
repeated after an interruption. ``--reuse-codebook`` starts from the published
codebook instead of training a new one, which removes the largest source of
numerical differences between machines.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MAIN = "frozen,context_only,single_rate,nested,full_online,nested_no_slots"
VARIANTS = [  # (tag, methods, overrides) - one seed each, on the hierarchical stream
    ("bits4", "nested", ["nested.sync.bits=4"]),
    ("lean", "nested", ["nested.sync.bits=4", "nested.sync.topk=0.1", "nested.periods.slow=100"]),
    ("biasonly", "nested", ["nested.lrs.slow=0.0"]),
    ("lr3x", "nested", ["nested.lrs.medium=2.0e-3", "nested.lrs.slow=1.5e-3"]),
    ("rayleigh", "nested", ["channel.type=rayleigh"]),
    ("rayleigh", "context_only,frozen", ["channel.type=rayleigh"]),
]


def plan(args) -> list[tuple[str, list[str], list[str]]]:
    """(label, script and options, overrides) for every step."""
    out = [f"out_dir={args.out}"]
    fixed = [f"out_dir={args.out}_fixed", f"codec.granularities=[{args.fixed_m}]", f"stream.m_op={args.fixed_m}"]
    steps = [
        ("corpora", ["prepare_data.py"], out),
        ("codebook", ["pretrain.py"], out),
        ("hierarchical stream, main methods, 3 seeds", ["run_stream.py", "--methods", MAIN, "--seeds", "0,1,2"], out + ["stream.kind=hier"]),
        ("hierarchical stream, ablations", ["run_stream.py", "--methods", "nested_no_context,nested_two_level,ewc", "--seeds", "0"], out + ["stream.kind=hier"]),
        ("temporal stream, 2 seeds", ["run_stream.py", "--methods", "frozen,context_only,single_rate,nested", "--seeds", "0,1"], out + ["stream.kind=temporal"]),
        ("temporal stream, upper reference", ["run_stream.py", "--methods", "full_online", "--seeds", "0"], out + ["stream.kind=temporal"]),
    ]
    for tag, methods, overrides in VARIANTS:
        steps.append((f"variant '{tag}' ({methods})", ["run_stream.py", "--methods", methods, "--seeds", "0", "--tag", tag], out + ["stream.kind=hier"] + overrides))
    steps += [
        ("memory headroom diagnostic", ["headroom.py"], out),
        ("selective refinement on the base corpus", ["eval_refine.py"], out),
        ("rate and fidelity on the base corpus", ["eval_rate.py"], out),
        (f"fixed-rate codebook ({args.fixed_m} symbols only)", ["pretrain.py"], fixed),
        ("tables", ["aggregate.py"], out),
        ("figures", ["make_figures.py"], out),
        ("export", ["export_results.py", "--to", args.export], out),
        ("report", ["write_report.py", "--to", args.export], out),
    ]
    return steps


def finished(label: str, command: list[str], overrides: list[str], args) -> bool:
    """Steps whose single output file tells that they are done (the stream
    and pretraining scripts skip finished work by themselves)."""
    eval_dir = Path(args.out) / "eval"
    if command[0] == "headroom.py":
        return (eval_dir / "headroom.json").exists()
    if command[0] == "eval_refine.py":
        return any(eval_dir.glob("refine_*.json"))
    if command[0] == "eval_rate.py":
        return any(eval_dir.glob("rate_*.json"))
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True, help="run directory for the reproduction")
    parser.add_argument("--export", required=True, help="folder for the exported tables, figures and compact results")
    parser.add_argument("--reference", default="", help="published results folder to compare with (e.g. results/gpt2)")
    parser.add_argument("--reuse-codebook", action="store_true", help="start from <reference>/base/codec.pt instead of training a codebook")
    parser.add_argument("--fixed-m", type=int, default=8, help="prefix length of the fixed-rate codebook check")
    parser.add_argument("--dry-run", action="store_true", help="print the steps without running them")
    parser.add_argument("overrides", nargs="*", help="config overrides passed to every step (key=value), e.g. device=cuda")
    args = parser.parse_args()

    if args.reuse_codebook and not args.dry_run:
        source = Path(args.reference) / "base" / "codec.pt"
        target = Path(args.out) / "base" / "codec.pt"
        if not args.reference or not source.exists():
            raise SystemExit("--reuse-codebook needs --reference with a base/codec.pt in it")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            for name in ("pretrain_summary.json", "pretrain_log.jsonl", "codec.pt"):  # the checkpoint last: it marks the step as done
                if (source.parent / name).exists():
                    shutil.copy2(source.parent / name, target.parent / name)
            print(f"[reproduce] using the published codebook {source}")

    steps = plan(args)
    started = time.time()
    for index, (label, command, overrides) in enumerate(steps, start=1):
        cmd = [sys.executable, "-u", str(ROOT / command[0]), "--config", args.config, *command[1:], *overrides, *args.overrides]
        shown = " ".join(["python", "scripts/" + command[0], "--config", args.config, *command[1:], *overrides, *args.overrides])
        print(f"\n===== [{index}/{len(steps)}] {label}  ({(time.time() - started) / 60:.0f} min elapsed)\n      {shown}", flush=True)
        if args.dry_run:
            continue
        if finished(label, command, overrides, args):
            print("      already done - skipped", flush=True)
            continue
        code = subprocess.run(cmd, cwd=ROOT.parent).returncode
        if code != 0:
            raise SystemExit(f"\nSTOPPED: step {index} ('{label}') failed with exit code {code}. Finished work is kept; fix the cause and run the same command again.")
    if args.dry_run:
        return

    summary = Path(f"{args.out}_fixed") / "base" / "pretrain_summary.json"
    if summary.exists():  # the fixed-rate codebook check is reported next to the main codebook
        target = Path(args.export) / f"fixed{args.fixed_m}"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(summary, target / "pretrain_summary.json")
    if args.reference:
        cmd = [sys.executable, str(ROOT / "compare_results.py"), "--reference", args.reference, "--new", args.export]
        print(f"\n===== comparison with {args.reference}", flush=True)
        subprocess.run(cmd, cwd=ROOT.parent)
    print(f"\nREPRODUCTION FINISHED in {(time.time() - started) / 60:.0f} min. Results: {args.export}")


if __name__ == "__main__":
    main()
