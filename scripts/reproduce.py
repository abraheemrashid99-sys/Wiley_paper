"""Run a whole study from scratch with one command and compare it with the published numbers.

    python scripts/reproduce.py --config configs/gpt2.yaml --out runs/gpt2_repro \
        --export results_repro/gpt2 --reference results/gpt2 --profile published device=auto

Profiles (each contains the one before it, with the same file names, so a
larger profile can always be compared with a smaller reference):

published  exactly the runs behind ``results/gpt2`` (version 0.2): the main
           methods on three seeds, the ablations and variants on one, with
           the decoder of those runs (greedy, shortest first shell, no
           reopening) whatever the configuration selects.
core       the confirmatory protocol of docs/METHOD.md, section 9: the main
           methods and the time-nesting ablation on six seeds on both
           streams, the other ablations and the Rayleigh stream on three.
paper      core plus the sensitivity studies a reviewer asks for (one seed
           each, defined relative to the configuration): shipping periods
           halved and doubled, a 2-bit quantiser, drift twice as fast and
           twice as slow, mean SNR 5 dB lower and higher, context length,
           adapter rank, and - when the temporal corpus is not already a
           stream domain - three domains against two and three adapter slots.

Steps, in order: data, speed benchmark, codebook, (core and paper: the
decoder selection repeated on validation text, as a check), the stream runs, the
diagnostics and evaluations on the base corpus, the fixed-rate codebook check;
then tables, figures, the exported folder with its report, and finally the
comparison with the reference folder (``COMPARISON.md`` in the export folder).

Everything that is already finished is skipped, so the command can simply be
repeated after an interruption. ``--reuse-codebook`` starts from the published
codebook instead of training a new one, which removes the largest source of
numerical differences between machines. ``--seeds`` overrides the seeds of the
main methods (e.g. ``--seeds 0,1,2,3,4,5,6,7,8,9``).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from dnsc.cli import utf8_console

ROOT = Path(__file__).resolve().parent
# every script started from here imports the package next to it, even if another copy is installed
CHILD_ENV = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT.parent), *([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])])}
sys.path.insert(0, str(ROOT.parent))  # the 'paper' profile reads the configuration with dnsc.config
PROFILES = ("published", "core", "paper")
MAIN = "frozen,context_only,single_rate,nested,full_online,nested_no_slots"
TEMPORAL = "frozen,context_only,single_rate,nested"
KEY = "context_only,single_rate,nested_no_slots,nested"  # the methods a changed stream is re-run with
# The decoder of selective refinement in the published runs (version 0.3 and earlier): greedy, first shell
# the shortest prefix, closed tokens never reopened. The first shell decides which tokens the emulated
# receiver writes into its memories, so the published numbers are reproduced with this decoder only.
LEGACY_DECODER = ["decoder.beam=1", "decoder.start_m=null", "decoder.reopen=false", "decoder.refine_power=message"]
# (tag, methods, overrides): variants of the hierarchical stream, one seed each in the published study
VARIANTS = [
    ("bits4", "nested", ["nested.sync.bits=4"]),
    ("lean", "nested", ["nested.sync.bits=4", "nested.sync.topk=0.1", "nested.periods.slow=100"]),
    ("biasonly", "nested", ["nested.lrs.slow=0.0"]),
    ("lr3x", "nested", ["nested.lrs.medium=2.0e-3", "nested.lrs.slow=1.5e-3"]),
    ("rayleigh", "nested", ["channel.type=rayleigh"]),
    ("rayleigh", "context_only,frozen", ["channel.type=rayleigh"]),
]


def sweeps(cfg) -> list[tuple[str, str, list[str]]]:
    """Sensitivity studies of the 'paper' profile (seed 0), defined relative
    to the configuration so that they mean the same for every backbone."""
    stream, nested = cfg["stream"], cfg["nested"]
    medium, slow = int(nested["periods"]["medium"]), int(nested["periods"]["slow"])
    tau_slow, tau_med, cycles = int(stream["tau_slow"]), int(stream["tau_med"]), int(stream["cycles"])
    mean_snr = float(stream["snr"]["mean"])
    rank, alpha = int(cfg["codec"]["lora"]["r"]), float(cfg["codec"]["lora"]["alpha"])
    context = int(cfg["context"]["max_tokens"])
    out = [
        # how often each level is shipped (the staleness / air-time trade-off of docs/ANALYSIS.md, section 4)
        ("slow_half", "nested", [f"nested.periods.slow={max(medium, slow // 2)}"]),
        ("slow_double", "nested", [f"nested.periods.slow={2 * slow}"]),
        ("medium_half", "nested", [f"nested.periods.medium={max(1, medium // 2)}"]),
        ("medium_double", "nested", [f"nested.periods.medium={min(slow, 2 * medium)}"]),
        ("bits2", "nested", ["nested.sync.bits=2"]),
        # drift speed: the same number of steps with regimes half and twice as long
        ("drift_fast", KEY, [f"stream.tau_slow={max(2, tau_slow // 2)}", f"stream.tau_med={max(1, tau_med // 2)}", f"stream.cycles={2 * cycles}"]),
        ("drift_slow", KEY, [f"stream.tau_slow={2 * tau_slow}", f"stream.tau_med={2 * tau_med}", f"stream.cycles={max(1, cycles // 2)}"]),
        # link quality: mean SNR 5 dB lower and higher
        ("snr_low", "frozen," + KEY, [f"stream.snr.mean={mean_snr - 5.0}"]),
        ("snr_high", "frozen," + KEY, [f"stream.snr.mean={mean_snr + 5.0}"]),
        # length of the free memory, size of the slow memory
        ("context_short", "context_only", [f"context.max_tokens={max(1, context // 3)}"]),
        ("context_long", "context_only", [f"context.max_tokens={2 * context}"]),
        ("rank_half", "nested", [f"codec.lora.r={max(1, rank // 2)}", f"codec.lora.alpha={alpha / 2:g}"]),
        ("rank_double", "nested", [f"codec.lora.r={2 * rank}", f"codec.lora.alpha={2 * alpha:g}"]),
    ]
    third = str(stream["temporal_corpus"])
    if third not in list(stream["domains"]):  # more domains than adapter slots, and one slot per domain
        domains = "[" + ",".join(list(stream["domains"]) + [third]) + "]"
        out.append(("three_domains", KEY, [f"stream.domains={domains}"]))
        out.append(("three_domains_three_slots", "nested", [f"stream.domains={domains}", f"codec.lora.slots={len(stream['domains']) + 1}"]))
    return out


def plan(args) -> list[tuple[str, list[str], list[str]]]:
    """(label, script and options, overrides) for every step."""
    confirmatory = args.profile in ("core", "paper")
    # the published runs used the version 0.3 decoder; the confirmatory runs use the configured one
    out = [f"out_dir={args.out}"] + ([] if confirmatory else LEGACY_DECODER)
    fixed = [f"out_dir={args.out}_fixed", f"codec.granularities=[{args.fixed_m}]", f"stream.m_op={args.fixed_m}"]
    seeds = args.seeds or ("0,1,2,3,4,5" if confirmatory else "0,1,2")
    temporal_seeds = args.seeds or ("0,1,2,3,4,5" if confirmatory else "0,1")
    few = "0,1,2" if confirmatory else "0"
    hier, temporal = out + ["stream.kind=hier"], out + ["stream.kind=temporal"]

    def stream(methods: str, which: str, *more: str) -> list[str]:
        return ["run_stream.py", "--methods", methods, "--seeds", which, *more]

    steps = [
        ("corpora", ["prepare_data.py"], out),
        ("speed benchmark (measured cost)", ["benchmark_speed.py"], out),
        ("codebook", ["pretrain.py"], out),
    ]
    if confirmatory:  # the selection of the decoder, repeated on validation text as a check of the configured one
        steps.append(("decoder selection on validation text (check)", ["tune_decoder.py"], out))
    steps += [
        (f"hierarchical stream, main methods, seeds {seeds}", stream(MAIN, seeds), hier),
    ]
    if confirmatory:
        steps.append((f"hierarchical stream, time-nesting ablation, seeds {seeds}", stream("nested_two_level", seeds), hier))
        steps.append((f"hierarchical stream, other ablations, seeds {few}", stream("nested_no_context,ewc", few), hier))
    else:
        steps.append(("hierarchical stream, ablations", stream("nested_no_context,nested_two_level,ewc", "0"), hier))
    steps += [
        (f"temporal stream, seeds {temporal_seeds}", stream(TEMPORAL, temporal_seeds), temporal),
        (f"temporal stream, upper reference, seeds {few}", stream("full_online", few), temporal),
    ]
    for tag, methods, overrides in VARIANTS:
        which = few if tag == "rayleigh" else "0"
        steps.append((f"variant '{tag}' ({methods})", stream(methods, which, "--tag", tag), hier + overrides))
    if args.profile == "paper":
        from dnsc.config import load_config

        for tag, methods, overrides in sweeps(load_config(ROOT.parent / args.config if not Path(args.config).is_absolute() else args.config, args.overrides)):
            steps.append((f"sensitivity '{tag}' ({methods})", stream(methods, "0", "--tag", tag), hier + overrides))
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
    if command[0] == "benchmark_speed.py":
        return (eval_dir / "speed.json").exists()
    if command[0] == "tune_decoder.py":
        path = eval_dir / "decoder_selection.json"
        return path.exists() and "decision" in path.read_text(encoding="utf-8")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True, help="run directory for the reproduction")
    parser.add_argument("--export", required=True, help="folder for the exported tables, figures and compact results")
    parser.add_argument("--reference", default="", help="published results folder to compare with (e.g. results/gpt2)")
    parser.add_argument("--profile", default="published", choices=PROFILES, help="which set of runs (see the top of this file)")
    parser.add_argument("--seeds", default="", help="seeds of the main methods on both streams (default: 0,1,2 published; 0..5 core and paper)")
    parser.add_argument("--reuse-codebook", action="store_true", help="start from <reference>/base/codec.pt instead of training a codebook")
    parser.add_argument("--fixed-m", type=int, default=8, help="prefix length of the fixed-rate codebook check")
    parser.add_argument("--dry-run", action="store_true", help="print the steps without running them")
    parser.add_argument("overrides", nargs="*", help="config overrides passed to every step (key=value), e.g. device=cuda; what defines a step (e.g. the setting a variant changes) takes precedence")
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
        # the caller's overrides come first, so that the settings which define a step (its output
        # folder, its stream, what a variant changes) are applied last and cannot be overridden
        cmd = [sys.executable, "-u", str(ROOT / command[0]), "--config", args.config, *command[1:], *args.overrides, *overrides]
        shown = " ".join(["python", "scripts/" + command[0], "--config", args.config, *command[1:], *args.overrides, *overrides])
        print(f"\n===== [{index}/{len(steps)}] {label}  ({(time.time() - started) / 60:.0f} min elapsed)\n      {shown}", flush=True)
        if args.dry_run:
            continue
        if finished(label, command, overrides, args):
            print("      already done - skipped", flush=True)
            continue
        code = subprocess.run(cmd, cwd=ROOT.parent, env=CHILD_ENV).returncode
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
        subprocess.run(cmd, cwd=ROOT.parent, env=CHILD_ENV)
    print(f"\nREPRODUCTION FINISHED in {(time.time() - started) / 60:.0f} min. Results: {args.export}")


if __name__ == "__main__":
    utf8_console()  # Windows consoles default to a legacy code page; the output contains ± and −
    main()
