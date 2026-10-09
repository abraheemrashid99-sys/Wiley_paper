"""Uncertainty-triggered incremental refinement of the base codec.

    python scripts/eval_refine.py --config configs/main.yaml --corpus europarl
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dnsc.cli import make_parser, parse
from dnsc.codec import load_codec
from dnsc.data import get_corpus
from dnsc.metrics import load_similarity
from dnsc.refine import run_refinement
from dnsc.utils import environment, get_device, save_json, set_seed


def main() -> None:
    parser = make_parser(__doc__)
    parser.add_argument("--corpus", default="", help="corpus (default: first base corpus)")
    parser.add_argument("--ckpt", default="", help="checkpoint directory (default: <out_dir>/base)")
    cfg, args = parse(parser)
    set_seed(int(cfg["seed"]))
    device = get_device(cfg["device"])
    corpus_name = args.corpus or cfg["data"]["base"][0]
    ckpt = Path(args.ckpt) if args.ckpt else Path(cfg["out_dir"]) / "base"
    codec = load_codec(cfg, ckpt, device)
    codec.eval()

    rcfg = cfg["refine"]
    corpus = get_corpus(corpus_name, cfg["data"])
    rng = np.random.default_rng(1)
    val, test = corpus.texts("val"), corpus.texts("test")
    val = [val[int(i)] for i in rng.permutation(len(val))[: int(rcfg["n_val"])]]
    test = [test[int(i)] for i in rng.permutation(len(test))[: int(rcfg["n_test"])]]
    sim = load_similarity(cfg["eval"]["sim_model"], device) if rcfg["quality"] == "sim" else None

    results = []
    for kind in cfg["eval"]["channels"]:
        for snr in cfg["eval"]["snrs"]:
            res = run_refinement(codec, val, test, float(snr), kind, rcfg, sim, int(cfg["eval"]["batch_size"]))
            results.append(res)
            best_fixed = max(res["fixed"], key=lambda r: r["quality"])
            print(
                f"[refine] {kind:<8} snr={snr:>4}  ECE raw={res['ece_raw']:.3f} -> calibrated={res['ece_calibrated']:.3f}  "
                f"oracle: {res['oracle']['symbols']:.1f} sym @ q={res['oracle']['quality']:.3f}  best fixed: m={best_fixed['m']} q={best_fixed['quality']:.3f}"
            )
    out = Path(cfg["out_dir"]) / "eval" / f"refine_{corpus_name}.json"
    save_json(out, {"corpus": corpus_name, "results": results, "config": cfg.to_dict(), "env": environment()})
    print(f"[refine] wrote {out}")


if __name__ == "__main__":
    main()
