"""Train the semantic codebook on the base corpora (minutes on a CPU).

    python scripts/pretrain.py --config configs/laptop.yaml
"""

from __future__ import annotations

from dnsc.cli import make_parser, parse
from dnsc.train import pretrain


def main() -> None:
    parser = make_parser(__doc__)
    parser.add_argument("--force", action="store_true", help="retrain even if a finished checkpoint exists")
    cfg, args = parse(parser)
    pretrain(cfg, force=args.force)


if __name__ == "__main__":
    main()
