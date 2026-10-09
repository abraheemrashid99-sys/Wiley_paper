"""Run a whole experiment pipeline with one command, on any operating system.

    python scripts/run_pipeline.py --config configs/smoke.yaml --stage smoke
    python scripts/run_pipeline.py --config configs/laptop.yaml --stage base
    python scripts/run_pipeline.py --config configs/laptop.yaml --stage streams --seeds 0,1,2
    python scripts/run_pipeline.py --config configs/laptop.yaml --stage report

Stages
------
smoke    everything, on the offline config (a few minutes)
base     environment check, data, speed benchmark, pretraining, rate and
         refinement evaluation
streams  both drift streams for the chosen methods and seeds
report   tables and figures from whatever results exist
all      base + streams + report

Every stage can be re-run: finished pretraining and finished stream runs are
skipped, and an interrupted pretraining resumes where it stopped.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_METHODS = "frozen,tx_only,single_rate,ewc,nested,nested_no_ema,nested_two_level"


def plan(stage: str, config: str, seeds: str, methods: str) -> list[list[str]]:
    base = [
        ["check_env.py"],
        ["prepare_data.py"],
        ["benchmark_speed.py"],
        ["pretrain.py"],
        ["eval_rate.py"],
        ["eval_refine.py"],
    ]
    streams = [
        ["run_stream.py", "--methods", methods, "--seeds", seeds, "stream.kind=hier"],
        ["run_stream.py", "--methods", methods, "--seeds", seeds, "stream.kind=temporal"],
    ]
    report = [["aggregate.py"], ["make_figures.py"]]
    smoke = [
        ["check_env.py"],
        ["prepare_data.py"],
        ["pretrain.py"],
        ["eval_rate.py"],
        ["run_stream.py", "--methods", "frozen,tx_only,single_rate,nested", "--seeds", "0,1"],
        ["eval_refine.py"],
        *report,
    ]
    steps = {"smoke": smoke, "base": base, "streams": streams, "report": report, "all": base + streams + report}[stage]
    commands = []
    for script, *rest in steps:
        options = [a for a in rest if "=" not in a or a.startswith("--")]
        overrides = [a for a in rest if "=" in a and not a.startswith("--")]
        commands.append([sys.executable, str(ROOT / script), "--config", config, *options, *overrides])
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=["smoke", "base", "streams", "report", "all"])
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    parser.add_argument("overrides", nargs="*", help="config overrides passed to every script (key=value)")
    args = parser.parse_args()

    commands = [cmd + list(args.overrides) for cmd in plan(args.stage, args.config, args.seeds, args.methods)]
    started = time.time()
    for index, cmd in enumerate(commands, start=1):
        shown = " ".join(["python", "scripts/" + Path(cmd[1]).name, *cmd[2:]])
        print(f"\n===== [{index}/{len(commands)}] {shown}", flush=True)
        if args.dry_run:
            continue
        code = subprocess.run(cmd, cwd=ROOT.parent).returncode
        if code != 0:
            raise SystemExit(f"\nSTOPPED: step {index} failed with exit code {code}. Fix the cause and run the same command again; finished work is kept.")
    if not args.dry_run:
        print(f"\nPIPELINE '{args.stage}' FINISHED in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
