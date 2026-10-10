"""Verify the machine before spending hours on it.

    python scripts/check_env.py --config configs/main.yaml

Checks, in order: library versions and device; that the backbone downloads
and runs one adaptation step and one decode on the selected device; that each
corpus the config references can be built; and that the sentence-similarity
model loads.
Each check prints PASS or FAIL with the reason, and the script exits non-zero
if anything failed.
"""

from __future__ import annotations

import time
import traceback

import torch

from dnsc.channel import Channel
from dnsc.cli import make_parser, parse
from dnsc.codec import build_codec
from dnsc.data import get_corpus
from dnsc.metrics import load_similarity
from dnsc.utils import count_params, environment, get_device


def check(label: str, fn) -> bool:
    start = time.perf_counter()
    try:
        detail = fn()
        print(f"PASS  {label}  ({time.perf_counter() - start:.1f}s)  {detail or ''}")
        return True
    except Exception as exc:
        print(f"FAIL  {label}: {exc!r}")
        traceback.print_exc(limit=3)
        return False


def main() -> None:
    parser = make_parser(__doc__)
    parser.add_argument("--skip-data", action="store_true", help="do not build the corpora")
    cfg, args = parse(parser)
    device = get_device(cfg["device"])
    print("environment:", environment())
    print("device:", device)
    results = []

    def backbone():
        codec = build_codec(cfg, device)
        groups = codec.groups()
        sizes = {name: count_params(p for _, p in params) for name, params in groups.items()}
        texts = ["The network slice is reconfigured before the handover.", "Parliament adopted the resolution on energy policy."]
        batch = codec.tokenize(texts)
        for _, param in groups["rx_lora"] + groups["rx_bias"]:
            param.requires_grad_(True)
        channel = Channel("awgn", 0)
        y, prec = codec.transmit(codec.encode(batch), batch.lens, 10.0, channel)
        ce, _, _ = codec.loss_at(batch, y, prec, codec.grans[0], codec.prior_tf(batch))
        ce.mean().backward()
        if not torch.isfinite(ce).all():
            raise RuntimeError("non-finite loss on this device")
        decoded = codec.reconstruct(texts[:1], codec.grans[-1], 20.0, Channel("awgn", 1)).texts[0]
        return f"hidden={codec.hidden} vocab={codec.vocab} groups={sizes} loss={ce.mean().item():.3f} decode_at_max_rate={decoded[:48]!r}"

    results.append(check(f"backbone '{cfg['backbone']['name']}' forward/backward/generate on {device}", backbone))

    if not args.skip_data:
        names = list(dict.fromkeys(list(cfg["data"]["base"]) + list(cfg["stream"]["domains"]) + [cfg["stream"]["temporal_corpus"]]))
        for name in names:
            results.append(check(f"corpus '{name}'", lambda name=name: str(get_corpus(name, cfg["data"]).summary())))

    if cfg["eval"]["sim_model"]:

        def similarity():
            sim = load_similarity(cfg["eval"]["sim_model"], device)
            if sim is None:
                raise RuntimeError("model could not be loaded")
            score = float(sim(["the cat sat on the mat"], ["a cat is sitting on the mat"])[0])
            return f"cosine(paraphrase)={score:.3f}"

        results.append(check(f"similarity model '{cfg['eval']['sim_model']}'", similarity))

    failed = results.count(False)
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
