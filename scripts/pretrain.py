"""Train the base (rate-nested) codec on the base corpora.

    python scripts/pretrain.py --config configs/laptop.yaml

The run is resumable. If it is interrupted (Ctrl+C, sleep, power loss), run
the same command again and it continues from the last saved state.
"""

from __future__ import annotations

from dnsc.cli import make_parser, parse
from dnsc.train import pretrain


def main() -> None:
    parser = make_parser(__doc__)
    parser.add_argument("--fresh", action="store_true", help="ignore a saved partial run and start from step 0")
    parser.add_argument("--force", action="store_true", help="retrain even if a finished checkpoint exists")
    cfg, args = parse(parser)
    pretrain(cfg, resume=not args.fresh, force=args.force or args.fresh)


if __name__ == "__main__":
    main()
