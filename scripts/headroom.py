"""How much is there to gain by adapting the receiver's prior? (diagnostic)

    python scripts/headroom.py --config configs/laptop.yaml

Measures, on held-out text of every domain, what each memory could buy at
best: the earlier sentences of the document as context, and a short adaptation
of the logit bias and of the LoRA adapter on top of it (including what that
adaptation does to the other domains). Run it after pretraining and before the long stream runs: it
tells in a few minutes whether there is anything worth adapting to.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dnsc.cli import make_parser, parse
from dnsc.headroom import format_rows, run_headroom


def main() -> None:
    parser = make_parser(__doc__)
    parser.add_argument("--ckpt", default="", help="checkpoint directory (default: <out_dir>/base)")
    parser.add_argument("--n", type=int, default=64, help="held-out sentences per domain")
    parser.add_argument("--steps", type=int, default=150, help="adaptation steps for the gradient memories")
    cfg, args = parse(parser)
    ckpt = Path(args.ckpt) if args.ckpt else Path(cfg["out_dir"]) / "base"
    out = Path(cfg["out_dir"]) / "eval" / "headroom.json"
    result = run_headroom(cfg, ckpt, out, n=args.n, steps=args.steps)
    frame = pd.DataFrame(format_rows(result))
    print(f"\nHeadroom of each memory at m={result['m_op']} symbols/token and {result['snr_db']:g} dB (oracle inputs; negative Δ bits = better)")
    print(frame.to_string(index=False))
    print(f"\n[headroom] wrote {out}")


if __name__ == "__main__":
    main()
