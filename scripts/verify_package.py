"""Check an unpacked release against its manifest.

    python scripts/verify_package.py

``MANIFEST.sha256`` (written by ``scripts/make_release.py``) lists the SHA-256
digest of every file of the release. This script recomputes them and reports
files that are missing or differ, so that a run on another machine starts
from exactly the code and reference results that were released. Files created
afterwards (data, models, runs, the virtual environment) are ignored.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            sha.update(chunk)
    return sha.hexdigest()


def main() -> None:
    manifest = ROOT / "MANIFEST.sha256"
    if not manifest.exists():
        raise SystemExit("MANIFEST.sha256 not found: this is a git checkout, not an unpacked release (nothing to verify)")
    missing, changed, total = [], [], 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split("  ", 1)
        total += 1
        path = ROOT / name
        if not path.exists():
            missing.append(name)
        elif digest(path) != expected:
            changed.append(name)
    revision = (ROOT / "REVISION").read_text(encoding="utf-8").strip() if (ROOT / "REVISION").exists() else "unknown"
    print(f"release: {revision}")
    print(f"{total - len(missing) - len(changed)}/{total} files match the manifest")
    for name in missing:
        print(f"  MISSING  {name}")
    for name in changed:
        print(f"  CHANGED  {name}")
    if missing or changed:
        raise SystemExit(1)
    print("PACKAGE VERIFIED: every file is exactly as released.")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    main()
