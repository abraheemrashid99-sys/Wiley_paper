"""Uncertainty-triggered selective refinement of the base codec.

    python scripts/eval_refine.py --config configs/laptop.yaml --corpus europarl

For every channel and SNR: a teacher-forced sweep of the rate-accuracy curve
with calibration error and the fixed-rate, oracle and random controls. At the
stream's mean SNR the receiver is additionally run for real (free-running,
multi-round) at a few thresholds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dnsc.channel import Channel
from dnsc.cli import make_parser, parse
from dnsc.codec import load_codec
from dnsc.data import get_corpus
from dnsc.metrics import load_similarity
from dnsc.refine import run_refinement, selective_decode
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
    sim = load_similarity(cfg["eval"]["sim_model"], device)
    batch_size = int(cfg["eval"]["batch_size"])
    live_snr = float(cfg["stream"]["snr"]["mean"])

    results = []
    for kind in cfg["eval"]["channels"]:
        for snr in sorted(set(float(s) for s in cfg["eval"]["snrs"]) | {live_snr}):
            res, calibrator = run_refinement(codec, val, test, snr, kind, batch_size)
            best_gain = max(res["curve"], key=lambda c: c["gain_over_fixed"])
            print(
                f"[refine] {kind:<8} snr={snr:>4}  token ECE raw={res['ece_raw']:.3f} -> calibrated={res['ece_calibrated']:.3f}  "
                f"best gain over fixed rate: +{best_gain['gain_over_fixed']:.3f} accuracy at {best_gain['symbols']:.2f} symbols/token  "
                f"(oracle: {res['oracle']['symbols']:.2f} sym @ {res['oracle']['quality']:.3f})"
            )
            if snr == live_snr:  # run the receiver for real at the pre-set thresholds
                res["free_running"] = []
                for tau in rcfg["live_thresholds"]:
                    live = selective_decode(codec, test[: int(rcfg["n_live"])], snr, Channel(kind, seed=4103), calibrator, tau, sim, batch_size)
                    res["free_running"].append(live)
                    print(f"[refine]   live threshold={tau:.3f}: {live['symbols']:.2f} symbols/token  token acc={live['gen_tok_acc']:.3f}  bleu4={live['bleu4']:.3f}  requests/token={live['requests_per_token']:.3f}")
            results.append(res)
    out = Path(cfg["out_dir"]) / "eval" / f"refine_{corpus_name}.json"
    save_json(out, {"corpus": corpus_name, "results": results, "config": cfg.to_dict(), "env": environment()})
    print(f"[refine] wrote {out}")


if __name__ == "__main__":
    main()
