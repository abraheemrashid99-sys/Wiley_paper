"""Rate-fidelity evaluation of the codec against its ablations and against
separation-based references.

For every channel, SNR and prefix length (symbols per token) this reports the
fidelity of Bayesian decoding under three priors - the LLM, a unigram model
and no prior at all - so the contribution of the language model is measured
directly. At the same symbol budget and under the same fading distribution it
also reports the delivery rate of ideal separate source-channel designs (see
dnsc/classical.py).

    python scripts/eval_rate.py --config configs/laptop.yaml --corpus europarl
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dnsc.channel import Channel
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

    n_tokens = np.array([int(codec.tokenize([t]).lens[0]) for t in texts], dtype=float)
    all_ids = codec.tokenize(texts)
    bits = {"sep_deflate": np.array([deflate_bits(t) for t in texts], dtype=float), "sep_lm": lm_bits(codec, texts, batch_size)}
    timer = Timer()
    rows, decoded, references = [], [], []
    for kind in ecfg["channels"]:
        for snr in ecfg["snrs"]:
            for m in codec.grans:
                shown = {}
                for prior in ecfg["priors"]:
                    row = {"system": f"dnsc_{prior}", "prior": prior, "channel": kind, "snr_db": float(snr), "m": int(m), "codebook_power": codec.codebook_power(all_ids, m)}
                    row.update(evaluate_tf(codec, texts, m, snr, Channel(kind, seed=2024), prior=prior, batch_size=batch_size))
                    gen = evaluate_gen(codec, texts, m, snr, Channel(kind, seed=2025), sim, prior=prior, batch_size=batch_size, return_texts=True)
                    references, hyps = gen.pop("refs"), gen.pop("hyps")
                    if prior == "llm":  # kept so that further text metrics can be computed without re-running
                        decoded.append({"channel": kind, "snr_db": float(snr), "m": int(m), "hyps": hyps})
                    row.update(gen)
                    rows.append(row)
                    shown[prior] = row
                # the separation references see fading drawn from the same distribution
                eff = Channel(kind, seed=2025).effective_snr_db(len(texts), snr).numpy()
                ref_ok = 0.0
                for name, b in bits.items():
                    for suffix, eps in (("", None), ("_fbl", FBL_EPS)):
                        ok = float(delivered(b, n_tokens * m, eff, eps).mean())
                        rows.append({"system": name + suffix, "channel": kind, "snr_db": float(snr), "m": int(m), "delivered": ok, "n": len(texts)})
                        if name + suffix == "sep_lm_fbl":
                            ref_ok = ok
                summary = "  ".join(f"{p}: bleu4={r['bleu4']:.3f} acc={r['gen_tok_acc']:.3f}" for p, r in shown.items())
                print(f"[rate] {kind:<8} snr={snr:>4} m={m:>3}  {summary}  | ideal separation (LM, finite block) delivered={ref_ok:.3f}")

    source = {name: {"mean_bits": float(b.mean()), "p95_bits": float(np.quantile(b, 0.95)), "bits_per_token": float(b.sum() / n_tokens.sum())} for name, b in bits.items()}
    out = Path(cfg["out_dir"]) / "eval" / f"rate_{corpus_name}.json"
    save_json(out, {"corpus": corpus_name, "n": len(texts), "mean_tokens": float(n_tokens.mean()), "rows": rows, "references": references, "decoded": decoded, "source_bits": source, "fbl_eps": FBL_EPS, "seconds": timer.elapsed(), "config": cfg.to_dict(), "env": environment()})
    print(f"[rate] wrote {out}")


if __name__ == "__main__":
    main()
