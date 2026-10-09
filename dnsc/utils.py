"""Small shared helpers: seeding, device selection, logging, provenance."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name: str = "auto") -> torch.device:
    """Resolve ``auto`` to MPS (Apple Silicon), then CUDA, then CPU."""
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def configure_threads(n: int) -> int:
    """Set the number of intra-op CPU threads (0 = leave the default)."""
    if int(n) > 0:
        torch.set_num_threads(int(n))
    return torch.get_num_threads()


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
        return None  # JSON has no NaN/inf
    return obj


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(obj), handle, indent=2)
    os.replace(tmp, path)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


class JsonlLogger:
    """Append-only JSON-lines log, flushed on every record."""

    def __init__(self, path: str | Path, append: bool = False):
        self.path = Path(path)
        ensure_dir(self.path.parent)
        self._handle = open(self.path, "a" if append else "w", encoding="utf-8")

    def log(self, **record: Any) -> None:
        self._handle.write(json.dumps(_jsonable(record)) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent,
            timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def environment() -> dict[str, Any]:
    """Fingerprint stored with every result file, so no number is orphaned."""
    import transformers

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "transformers": transformers.__version__,
        "git": git_revision(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def count_params(params) -> int:
    return int(sum(p.numel() for p in params))


class Timer:
    def __init__(self) -> None:
        self.start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.start
