"""Copy the results of a run directory into a small folder that can be committed.

    python scripts/export_results.py --config configs/gpt2.yaml --to results/gpt2

Writes the tables and figures as they are, the evaluation files without the
decoded texts, one compact JSON per stream run (summary, operating curves,
probes, per-regime means and a 10-step-averaged trace instead of the full
per-step log) and the trained codebook checkpoint (without the tokenizer,
which comes with the backbone). The full per-step logs are
packed into ``raw_streams.tar.gz`` so that ``aggregate.py`` and
``make_figures.py`` can be re-run from the exported folder after unpacking
it into ``<out_dir>/stream``.
"""

from __future__ import annotations

import shutil
import tarfile
from pathlib import Path

import numpy as np

from dnsc.cli import make_parser, parse
from dnsc.utils import ensure_dir, load_json, save_json

TRACE = ("prior_bits", "tok_acc", "ce_bits", "snr_db")


def compact_stream(result: dict, every: int = 10) -> dict:
    log = result["log"]
    n = result["n_steps"]
    edges = list(range(0, n, every))
    trace = {key: [float(np.mean(log[key][a : a + every])) for a in edges] for key in TRACE}
    k_tau = [p["threshold"] for p in result["selective_curve"]].index(result["threshold"])
    trace["sel_symbols"] = [float(np.mean([row[k_tau] for row in log["sel_symbols"][a : a + every]])) for a in edges]
    trace["sel_acc"] = [float(np.mean([row[k_tau] for row in log["sel_acc"][a : a + every]])) for a in edges]
    trace["slot"] = [int(max(set(log["slot"][a : a + every]), key=log["slot"][a : a + every].count)) for a in edges]
    regimes = []
    regime = np.asarray(log["regime"])
    for index in sorted(set(log["regime"])):
        pick = regime == index
        regimes.append(
            {
                "regime": int(index),
                "label": log["domain"][int(np.argmax(pick))],
                "steps": int(pick.sum()),
                "prior_bits": float(np.asarray(log["prior_bits"])[pick].mean()),
                "tok_acc": float(np.asarray(log["tok_acc"])[pick].mean()),
                "sel_symbols": float(np.asarray(log["sel_symbols"])[pick][:, k_tau].mean()),
                "sel_acc": float(np.asarray(log["sel_acc"])[pick][:, k_tau].mean()),
            }
        )
    keep = ("method", "seed", "stream", "m_op", "n_steps", "context", "cache", "slots", "slot_switches", "rule", "threshold", "target_acc", "fixed_curve", "selective_curve", "levels", "summary", "probes", "config", "env")
    out = {key: result[key] for key in keep if key in result}
    out["ledger"] = {key: value for key, value in result["ledger"].items() if key != "cumulative"}
    out["regimes"] = regimes
    out["trace_every"] = every
    out["trace"] = trace
    return out


def main() -> None:
    parser = make_parser(__doc__)
    parser.add_argument("--to", required=True, help="destination folder")
    cfg, args = parse(parser)
    src, dst = Path(cfg["out_dir"]), ensure_dir(Path(args.to))

    for name in ("tables", "figures"):
        if (src / name).exists():
            shutil.copytree(src / name, dst / name, dirs_exist_ok=True)
    for path in sorted((src / "eval").glob("*.json")) if (src / "eval").exists() else []:
        data = load_json(path)
        for bulky in ("decoded", "references"):  # decoded texts stay in the run directory
            data.pop(bulky, None)
        save_json(ensure_dir(dst / "eval") / path.name, data)
    streams = sorted((src / "stream").glob("*.json")) if (src / "stream").exists() else []
    for path in streams:
        save_json(ensure_dir(dst / "stream") / path.name, compact_stream(load_json(path)))
    if streams:
        with tarfile.open(dst / "raw_streams.tar.gz", "w:gz") as archive:
            for path in streams:
                archive.add(path, arcname=path.name)
    base = src / "base"
    for name in ("codec.pt", "pretrain_summary.json", "pretrain_log.jsonl"):
        if (base / name).exists():
            shutil.copy2(base / name, ensure_dir(dst / "base") / name)
    print(f"[export] {len(streams)} stream runs, tables, figures and evaluations copied to {dst}")


if __name__ == "__main__":
    main()
