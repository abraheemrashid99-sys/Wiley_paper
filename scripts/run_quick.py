"""The primary comparison of the paper on a laptop, in hours, with one command.

    python scripts/run_quick.py                     # seeds 3,4,5 (default)
    python scripts/run_quick.py --seeds 0,1,2       # other seeds
    python scripts/run_quick.py --report-only       # tables from whatever has finished

What it runs: the hierarchical drift stream with the receivers of the primary
endpoint - ``frozen`` (the reference), ``nested`` (the proposal) and
``context_only`` (the free memory alone) - with the configured decoder
(version 0.4: beam 4, first round of 6 symbols) and the published codebook,
exactly as ``scripts/run_stream.py`` would; several runs in parallel, the
reference and the proposal of every seed first. Nothing is changed in the
method or the measurement: it is ``--profile core`` restricted to the runs
the primary endpoint needs, on the CPU.

Every finished run is kept, so the command can be stopped (Ctrl+C) and
started again; an interrupted run starts over. At the end, or with
``--report-only`` at any time, it writes the tables
(``<out>/tables/realdecode_hier.md`` is the primary endpoint), the report
``results_quick/TABLES.md``, and ``quick_results.zip`` - the raw runs and
the report - to send on or to pool with runs made elsewhere
(``--add-runs <folder or .zip/.tar.gz of stream/*.json>``; runs of the same
seed and method already present are kept).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dnsc.cli import utf8_console  # noqa: E402

METHODS = ("frozen", "nested", "context_only")  # the reference and the proposal of a seed come first


def memory_gb() -> float | None:
    """Total physical memory, without third-party packages (None if unknown)."""
    try:
        if sys.platform == "win32":
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

            status = Status()
            status.dwLength = ctypes.sizeof(Status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return status.ullTotalPhys / 1e9
        if sys.platform == "darwin":
            return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / 1e9
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except Exception:
        return None


def say(text: str) -> None:
    print(f"[quick {time.strftime('%H:%M:%S')}] {text}", flush=True)


def run(args: list[str], log: Path | None = None) -> None:
    cmd = [sys.executable, *args]
    if log is None:
        done = subprocess.run(cmd, cwd=ROOT)
    else:
        with open(log, "w", encoding="utf-8") as handle:
            done = subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if done.returncode != 0:
        raise SystemExit(f"failed: {' '.join(args)}" + (f" (log: {log})" if log else ""))


def last_step(log: Path) -> int:
    """The last step a running stream printed in its log (progress every 50 steps)."""
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return 0
    for line in reversed(lines):
        if " step " in line and "/" in line:
            try:
                return int(line.split(" step ", 1)[1].split("/", 1)[0])
            except ValueError:
                pass
    return 0


def add_runs(source: Path, stream_dir: Path) -> int:
    """Copy raw stream results made elsewhere into the stream folder."""
    added = 0

    def take(name: str, data: bytes) -> None:
        nonlocal added
        target = stream_dir / Path(name).name
        if target.exists():
            return
        result = json.loads(data)
        if "live" not in result or "log" not in result:
            say(f"not a raw stream result (no per-step logs), skipped: {name}")
            return
        target.write_bytes(data)
        added += 1

    if source.is_dir():
        for path in sorted(source.rglob("*.json")):
            take(path.name, path.read_bytes())
    elif source.suffix == ".zip":
        with zipfile.ZipFile(source) as archive:
            for name in archive.namelist():
                if name.endswith(".json") and "/stream/" in "/" + name:
                    take(name, archive.read(name))
    elif source.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(source) as archive:
            for member in archive.getmembers():
                if member.isfile() and member.name.endswith(".json"):
                    take(member.name, archive.extractfile(member).read())
    else:
        raise SystemExit(f"--add-runs: not a folder, .zip or .tar.gz: {source}")
    return added


def report(cfg_args: list[str], out: Path, export: Path) -> None:
    say("tables, figures and report ...")
    run(["scripts/aggregate.py", *cfg_args], out / "logs" / "aggregate.log")
    run(["scripts/make_figures.py", *cfg_args], out / "logs" / "figures.log")
    run(["scripts/export_results.py", *cfg_args, "--to", str(export)], out / "logs" / "export.log")
    run(["scripts/write_report.py", *cfg_args, "--to", str(export)], out / "logs" / "report.log")
    endpoint = out / "tables" / "realdecode_hier.md"
    if endpoint.exists():
        print("\n" + endpoint.read_text(encoding="utf-8"))
    check = out / "tables" / "emulation_check_hier.md"
    if check.exists():
        print(check.read_text(encoding="utf-8"))
    package = ROOT / "quick_results.zip"
    with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((out / "stream").glob("*.json")):
            archive.write(path, f"stream/{path.name}")
        for path in sorted(export.rglob("*")):
            if path.is_file() and path.suffix != ".gz":
                archive.write(path, f"report/{path.relative_to(export).as_posix()}")
    say(f"report: {export / 'TABLES.md'}")
    say(f"everything to send: {package} ({package.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    utf8_console()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/gpt2.yaml")
    parser.add_argument("--seeds", default="3,4,5", help="seeds of the stream (default 3,4,5: never run before)")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--out", default="runs/quick", help="run folder (default runs/quick)")
    parser.add_argument("--export", default="results_quick", help="report folder (default results_quick)")
    parser.add_argument("--ckpt", default="results/gpt2/base", help="codebook (default: the published one)")
    parser.add_argument("--workers", type=int, default=0, help="runs in parallel (default: half the logical CPUs, at most 4)")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads per run (default: logical CPUs / workers)")
    parser.add_argument("--add-runs", default="", help="raw stream results made elsewhere to pool with these (folder, .zip or .tar.gz)")
    parser.add_argument("--report-only", action="store_true", help="only write the tables from the runs that have finished")
    parser.add_argument("--dry-run", action="store_true", help="print what would run")
    parser.add_argument("overrides", nargs="*", help="key=value settings passed to every script (as for the other scripts)")
    args = parser.parse_args()

    out, export = ROOT / args.out, ROOT / args.export
    cfg_args = ["--config", args.config, f"out_dir={args.out}", "stream.kind=hier", *args.overrides]
    cpus = os.cpu_count() or 2
    memory = memory_gb()
    # a GPT-2 run needs about 2 GB; leave room for the system
    workers = args.workers or max(1, min(4, cpus // 2, int((memory - 3) // 2.2) if memory else 2))
    threads = args.threads or max(1, cpus // workers)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    jobs = [(m, s) for s in seeds for m in methods if m in ("frozen", "nested")] + [(m, s) for s in seeds for m in methods if m not in ("frozen", "nested")]
    stream_dir = out / "stream"
    for folder in (stream_dir, out / "logs"):
        folder.mkdir(parents=True, exist_ok=True)
    result = lambda m, s: stream_dir / f"hier_{m}_s{s}.json"  # noqa: E731
    todo = [(m, s) for m, s in jobs if not result(m, s).exists()]
    say(f"{len(jobs)} runs ({', '.join(methods)} x seeds {','.join(map(str, seeds))}), {len(jobs) - len(todo)} already done; {workers} in parallel x {threads} threads ({cpus} logical CPUs, {f"{memory:.0f} GB" if memory else "unknown"} memory)")

    if args.dry_run:
        for m, s in todo:
            print(f"    python scripts/run_stream.py --methods {m} --seeds {s} --ckpt {args.ckpt} {' '.join(cfg_args)} threads={threads}")
        return
    if args.add_runs:
        say(f"added {add_runs(Path(args.add_runs), stream_dir)} runs from {args.add_runs}")

    if not args.report_only and todo:
        # what every run needs: the language model and the two corpora of the stream
        cfg_probe = json.loads(subprocess.run([sys.executable, "-c", "import json,sys; from dnsc.config import load_config; c=load_config(sys.argv[1], sys.argv[2:]); print(json.dumps({'backbone': c['backbone']['name'], 'domains': list(c['stream']['domains'])}))", args.config, *args.overrides], cwd=ROOT, capture_output=True, text=True, check=True).stdout)
        backbone = ROOT / cfg_probe["backbone"]
        if cfg_probe["backbone"] == "models/gpt2" and not (backbone / "model.safetensors").exists():
            say("fetching GPT-2 (once) ...")
            run(["scripts/fetch_gpt2.py", "--out", "models/gpt2"])
        say("corpora (once) ...")
        run(["scripts/prepare_data.py", *cfg_args, "--corpora", ",".join(cfg_probe["domains"])], out / "logs" / "prepare_data.log")
        if not (ROOT / args.ckpt / "codec.pt").exists():
            raise SystemExit(f"no codebook at {args.ckpt}/codec.pt")

        running: dict[tuple[str, int], tuple[subprocess.Popen, float, Path]] = {}
        handles = []
        started, durations = time.time(), []
        last_note = 0.0
        try:
            while todo or running:
                while todo and len(running) < workers:
                    m, s = todo.pop(0)
                    log = out / "logs" / f"hier_{m}_s{s}.log"
                    handle = open(log, "w", encoding="utf-8")  # noqa: SIM115 (closed at the end)
                    handles.append(handle)
                    cmd = [sys.executable, "-u", "scripts/run_stream.py", "--methods", m, "--seeds", str(s), "--ckpt", args.ckpt, *cfg_args, f"threads={threads}"]  # key=value settings last
                    running[(m, s)] = (subprocess.Popen(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT), time.time(), log)
                    say(f"started {m} seed {s} (log: {os.path.relpath(log, ROOT) if log.is_relative_to(ROOT) else log})")
                time.sleep(5)
                for key, (proc, t0, log) in list(running.items()):
                    if proc.poll() is None:
                        continue
                    del running[key]
                    if proc.returncode != 0 or not result(*key).exists():
                        raise SystemExit(f"run {key[0]} seed {key[1]} failed - see {log}")
                    durations.append(time.time() - t0)
                    say(f"finished {key[0]} seed {key[1]} in {durations[-1] / 60:.0f} min")
                if time.time() - last_note > 600 and running:  # every 10 minutes: where every run is
                    last_note = time.time()
                    parts = [f"{m} s{s}: step {last_step(log)}/600" for (m, s), (_, _, log) in running.items()]
                    per_run = (sum(durations) / len(durations)) if durations else None
                    eta = "" if per_run is None else f"; about {per_run * (len(todo) + len(running) * 0.5) / workers / 60:.0f} min left"
                    say(" | ".join(parts) + f"; {len(todo)} waiting{eta} ({(time.time() - started) / 60:.0f} min so far)")
        except KeyboardInterrupt:
            for proc, _, _ in running.values():
                proc.terminate()
            say("stopped; finished runs are kept - run the same command to continue")
            return
        finally:
            for handle in handles:
                handle.close()
    report(cfg_args, out, export)


if __name__ == "__main__":
    main()
