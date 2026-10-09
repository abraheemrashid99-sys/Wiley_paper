"""Rate-fidelity evaluation of the base codec against separation-based references.

For every channel, SNR and prefix length this reports teacher-forced and
free-running fidelity of the semantic codec and, at the *same symbol budget*
and under the same fading distribution, the delivery rate of two ideal
separate source-channel designs (see dnsc/classical.py).

    python scripts/eval_rate.py --config configs/main.yaml --corpus europarl
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dnsc.channel import Channel, capacity_real
from dnsc.classical import deflate_bits, delivered, lm_bits
from dnsc.cli import make_parser, parse
from dnsc.codec import load_codec
from dnsc.data import get_corpus
from dnsc.metrics import load_similarity
from dnsc.train import evaluate_gen, evaluate_tf
from dnsc.utils import Timer, environment, get_device, save_json, set_seed


FBL_EPS = 1e-3  # block error rate of the finite-block-length reference


def main() -> None:
    parser = make_parser(__doc__)
    parser.add_argument("--corpus", default="", help="corpus to evaluate on (default: first base corpus)")
    parser.add_argument("--ckpt", default="", help="checkpoint directory (default: <out_dir>/base)")
    cfg, args = parse(parser)
    set_seed(int(cfg["seed"]))
    device = get_device(cfg["device"])
    corpus_name = args.corpus or cfg["data"]["base"][0]
    ckpt = Path(args.ckpt) if args.ckpt else Path(cfg["out_dir"]) / "base"

    codec = load_codec(cfg, ckpt, device)
    codec.eval()
    ecfg = cfg["eval"]
    texts = get_corpus(corpus_name, cfg["data"]).texts("test")
    rng = np.random.default_rng(0)
    texts = [texts[int(i)] for i in rng.permutation(len(texts))[: int(ecfg["n_test"])]]
    sim = load_similarity(ecfg["sim_model"], device)
    batch_size = int(ecfg["batch_size"])

    bits = {"sep_deflate": np.array([deflate_bits(t) for t in texts], dtype=float), "sep_lm": lm_bits(codec, texts, batch_size)}
    timer = Timer()
    rows = []
    for kind in ecfg["channels"]:
        for snr in ecfg["snrs"]:
            for m in codec.grans:
                row = {"system": "dnsc", "channel": kind, "snr_db": float(snr), "m": int(m)}
                row.update(evaluate_tf(codec, texts, m, snr, Channel(kind, seed=2024), batch_size))
                row.update(evaluate_gen(codec, texts, m, snr, Channel(kind, seed=2025), sim, batch_size))
                rows.append(row)
                # the separation references see fading drawn from the same distribution
                eff = Channel(kind, seed=2025).effective_snr_db(len(texts), snr).numpy()
                for name, b in bits.items():
                    for suffix, eps in (("", None), ("_fbl", FBL_EPS)):
                        ok = float(delivered(b, m, eff, eps).mean())
                        rows.append({"system": name + suffix, "channel": kind, "snr_db": float(snr), "m": int(m), "delivered": ok, "n": len(texts)})
                print(f"[rate] {kind:<8} snr={snr:>4} m={m:>4}  dnsc bleu4={row['bleu4']:.3f} tok_acc={row['tok_acc']:.3f}  ideal separation (LM, finite block) delivered={rows[-1]['delivered']:.3f}")

    source = {name: {"mean_bits": float(b.mean()), "p95_bits": float(np.quantile(b, 0.95))} for name, b in bits.items()}
    for name, stats in source.items():  # symbols an ideal code needs for 95 % delivery on AWGN
        stats["awgn_symbols_p95"] = {str(s): float(stats["p95_bits"] / capacity_real(float(s))) for s in ecfg["snrs"]}
    out = Path(cfg["out_dir"]) / "eval" / f"rate_{corpus_name}.json"
    save_json(out, {"corpus": corpus_name, "n": len(texts), "rows": rows, "source_bits": source, "fbl_eps": FBL_EPS, "seconds": timer.elapsed(), "config": cfg.to_dict(), "env": environment()})
    print(f"[rate] wrote {out}")


if __name__ == "__main__":
    main()
