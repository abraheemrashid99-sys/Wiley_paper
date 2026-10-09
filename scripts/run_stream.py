"""Online adaptation on a multi-timescale drift stream.

    python scripts/run_stream.py --config configs/main.yaml --methods frozen,tx_only,single_rate,nested --seeds 0,1,2
    python scripts/run_stream.py --config configs/main.yaml --methods nested --seeds 0 stream.kind=temporal

Existing result files are skipped unless --force is given, so an interrupted
sweep can simply be restarted.
"""

from __future__ import annotations

from pathlib import Path

from dnsc.cli import int_list, make_parser, parse, str_list
from dnsc.online import METHODS, run_stream


def main() -> None:
    parser = make_parser(__doc__)
    parser.add_argument("--methods", default="frozen,tx_only,single_rate,nested", help=f"comma-separated subset of {METHODS}")
    parser.add_argument("--seeds", default="0", help="comma-separated seeds")
    parser.add_argument("--ckpt", default="", help="checkpoint directory (default: <out_dir>/base)")
    parser.add_argument("--tag", default="", help="suffix for the result files (e.g. a sweep label)")
    parser.add_argument("--force", action="store_true", help="recompute even if the result file exists")
    cfg, args = parse(parser)
    ckpt = Path(args.ckpt) if args.ckpt else Path(cfg["out_dir"]) / "base"
    out_dir = Path(cfg["out_dir"]) / "stream"
    tag = f"_{args.tag}" if args.tag else ""
    for method in str_list(args.methods):
        for seed in int_list(args.seeds):
            out = out_dir / f"{cfg['stream']['kind']}_{method}_s{seed}{tag}.json"
            if out.exists() and not args.force:
                print(f"[stream] skip existing {out}")
                continue
            run_stream(cfg, method, seed, ckpt, out)


if __name__ == "__main__":
    main()
