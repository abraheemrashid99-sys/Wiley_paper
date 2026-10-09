"""Train the base (rate-nested) codec on the base corpora.

    python scripts/pretrain.py --config configs/main.yaml
"""

from __future__ import annotations

from dnsc.cli import make_parser, parse
from dnsc.train import pretrain


def main() -> None:
    cfg, _ = parse(make_parser(__doc__))
    pretrain(cfg)


if __name__ == "__main__":
    main()
