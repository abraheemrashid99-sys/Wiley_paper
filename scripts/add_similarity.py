"""Add sentence-embedding similarity to a rate evaluation, without re-running it.

    python scripts/add_similarity.py --results results/gpt2
    python scripts/add_similarity.py --results runs/gpt2 --model sentence-transformers/all-MiniLM-L6-v2

``scripts/eval_rate.py`` stores every decoded sentence (in the run directory
inside ``eval/rate_<corpus>.json``; in an exported folder in
``eval/rate_<corpus>.texts.json.gz``). This script embeds references and
reconstructions with a small sentence encoder and writes the mean cosine
similarity of every (prior, channel, SNR, prefix length) into the ``sim``
field of the matching row, so ``aggregate.py`` tabulates it like any other
metric. It needs the encoder from Hugging Face; where that is unreachable the
script says so and changes nothing.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from dnsc.metrics import load_similarity
from dnsc.utils import load_json, save_json

from dnsc.cli import utf8_console


def load_texts(path: Path) -> dict | None:
    """Decoded texts of one rate file: inside it (run directory) or in the
    compressed side file (exported folder)."""
    data = load_json(path)
    if data.get("decoded"):
        return {"decoded": data["decoded"], "references": data["references"]}
    side = path.with_name(path.stem + ".texts.json.gz")
    if side.exists():
        with gzip.open(side, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True, help="run directory or exported results folder")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    files = sorted((Path(args.results) / "eval").glob("rate_*.json"))
    if not files:
        raise SystemExit(f"no rate evaluation under {args.results}/eval")
    sim = load_similarity(args.model, torch.device(args.device))
    if sim is None:
        raise SystemExit("the sentence encoder could not be loaded (is Hugging Face reachable?); nothing was changed")
    for path in files:
        texts = load_texts(path)
        if texts is None:
            print(f"[sim] {path}: no decoded texts stored - re-run scripts/eval_rate.py")
            continue
        data = load_json(path)
        references = texts["references"]
        reference_vectors = sim.embed(references)
        index = {(row["prior"], row["channel"], float(row["snr_db"]), int(row["m"])): row for row in data["rows"] if row.get("prior")}
        done = 0
        for item in texts["decoded"]:
            row = index.get((item["prior"], item["channel"], float(item["snr_db"]), int(item["m"])))
            if row is None or len(item["hyps"]) != len(references):
                continue
            row["sim"] = float(np.mean((reference_vectors * sim.embed(item["hyps"])).sum(-1).numpy()))
            done += 1
        data["sim_model"] = args.model
        save_json(path, data)
        print(f"[sim] {path}: similarity added to {done} rows ({args.model})")


if __name__ == "__main__":
    utf8_console()  # Windows consoles default to a legacy code page; the output contains ± and −
    main()
