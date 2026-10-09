"""Shared command-line plumbing for the scripts."""

from __future__ import annotations

import argparse

from .config import Cfg, load_config


def make_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", required=True, help="path to a YAML config (see configs/)")
    parser.add_argument("overrides", nargs="*", help="config overrides of the form section.key=value")
    return parser


def parse(parser: argparse.ArgumentParser) -> tuple[Cfg, argparse.Namespace]:
    args = parser.parse_args()
    return load_config(args.config, args.overrides), args


def int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def str_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]
