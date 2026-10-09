"""Shared command-line plumbing for the scripts."""

from __future__ import annotations

import argparse
import sys

from .config import Cfg, load_config


def make_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", required=True, help="path to a YAML config (see configs/)")
    parser.add_argument("overrides", nargs="*", help="config overrides of the form section.key=value")
    return parser


def utf8_console() -> None:
    """Windows consoles and pipes default to a legacy code page; table output
    contains characters such as the plus-minus sign, so force UTF-8."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def parse(parser: argparse.ArgumentParser) -> tuple[Cfg, argparse.Namespace]:
    from .utils import configure_threads

    utf8_console()
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    configure_threads(int(cfg["threads"]))
    return cfg, args


def int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def str_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]
