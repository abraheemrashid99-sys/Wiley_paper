"""Build the release archive from the committed state of the repository.

    python scripts/make_release.py --out dist

Only files tracked by git at HEAD go into the archive (``git archive``), so
it can never contain a stray local file. Two files are added: ``REVISION``
(the commit the archive was built from; results produced from the unpacked
archive record it) and ``MANIFEST.sha256`` (the digest of every file, checked
by ``scripts/verify_package.py``). Line endings are stored as in the
repository. The command refuses to run with uncommitted changes.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import subprocess
import tarfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="dist")
    parser.add_argument("--allow-dirty", action="store_true", help="build although tracked files are modified (the archive still holds HEAD)")
    args = parser.parse_args()
    if git("status", "--porcelain", "--untracked-files=no") and not args.allow_dirty:
        raise SystemExit("tracked files have uncommitted changes: commit first, so that the archive and its REVISION agree")
    revision = git("rev-parse", "--short", "HEAD")
    version = next(line.split('"')[1] for line in (ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines() if line.startswith("version"))
    folder = f"DNSC_v{version}"
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{folder}_{revision}.zip"

    raw = subprocess.run(["git", "archive", "--format=tar", "HEAD"], cwd=ROOT, capture_output=True, check=True).stdout
    files: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for member in archive.getmembers():
            if member.isfile():
                files[member.name] = archive.extractfile(member).read()
    stamp = git("log", "-1", "--format=%cI")
    files["REVISION"] = f"{revision} (version {version}, commit of {stamp})\n".encode()
    manifest = "".join(f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in sorted(files.items()))
    files["MANIFEST.sha256"] = manifest.encode()

    date = time.strptime(stamp[:19], "%Y-%m-%dT%H:%M:%S")[:6]  # file times = commit time: the archive is reproducible
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(f"{folder}/{name}", date_time=date)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    print(f"{target}  ({target.stat().st_size / 1e6:.1f} MB, {len(files)} files)\nsha256 {digest}")


if __name__ == "__main__":
    main()
