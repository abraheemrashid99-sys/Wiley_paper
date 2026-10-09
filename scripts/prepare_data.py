"""Download, filter, split and cache every corpus a config needs.

    python scripts/prepare_data.py --config configs/main.yaml
"""

from __future__ import annotations

from pathlib import Path

from dnsc.cli import make_parser, parse
from dnsc.data import get_corpus
from dnsc.utils import save_json


def main() -> None:
    parser = make_parser(__doc__)
    parser.add_argument("--refresh", action="store_true", help="rebuild the cache even if it exists")
    parser.add_argument("--corpora", default="", help="comma-separated list; default: everything the config references")
    cfg, args = parse(parser)

    names = [n.strip() for n in args.corpora.split(",") if n.strip()]
    if not names:
        names = list(cfg["data"]["base"]) + list(cfg["stream"]["domains"]) + [cfg["stream"]["temporal_corpus"]]
    names = list(dict.fromkeys(names))

    rows, failed = [], []
    for name in names:
        try:
            rows.append(get_corpus(name, cfg["data"], refresh=args.refresh).summary())
        except Exception as exc:
            failed.append((name, repr(exc)))
            print(f"[data] FAILED to build '{name}': {exc!r}")

    if rows:
        keys = list(rows[0])
        widths = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in keys}
        print("\n" + "  ".join(k.ljust(widths[k]) for k in keys))
        for row in rows:
            print("  ".join(str(row[k]).ljust(widths[k]) for k in keys))
        save_json(Path(cfg["data"]["cache_dir"]) / "summary.json", rows)
    if failed:
        raise SystemExit(f"\n{len(failed)} corpus/corpora failed: {[n for n, _ in failed]}")
    print("\n[data] all corpora ready")


if __name__ == "__main__":
    main()
