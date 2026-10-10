"""Minimal LoRA with named adapters.

The backbone stays frozen; each wrapped linear layer holds one low-rank adapter
per named *slot* ("s0", "s1", ...), of which at most one is active at a time.
The slots are the receiver's slow memory: one adapter per knowledge domain.
Keeping every adapter in its own parameter tensors is what lets it be counted,
quantised and synchronised on its own.
"""

from __future__ import annotations

import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

_LAYER_INDEX = re.compile(r"\.(?:layers|h|blocks)\.(\d+)\.")


def _is_conv1d(module: nn.Module) -> bool:
    """GPT-2 style ``Conv1D``: a linear layer whose weight is stored (in, out)."""
    return module.__class__.__name__ == "Conv1D" and getattr(module, "weight", None) is not None and module.weight.dim() == 2


def _in_out(base: nn.Module) -> tuple[int, int]:
    if isinstance(base, nn.Linear):
        return base.in_features, base.out_features
    if _is_conv1d(base):
        return int(base.weight.shape[0]), int(base.weight.shape[1])
    raise TypeError(f"cannot attach LoRA to {type(base).__name__}")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Module, r: int, alpha: float, dropout: float, sides: tuple[str, ...]):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        dtype = base.weight.dtype
        n_in, n_out = _in_out(base)
        for side in sides:
            a = torch.empty(r, n_in, dtype=dtype)
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            self.lora_A[side] = nn.Parameter(a)
            self.lora_B[side] = nn.Parameter(torch.zeros(n_out, r, dtype=dtype))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.active: str | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        side = self.active
        if side is None or side not in self.lora_A:
            return out
        update = F.linear(F.linear(self.drop(x), self.lora_A[side]), self.lora_B[side])
        return out + self.scale * update


def _layer_index(name: str) -> int | None:
    match = _LAYER_INDEX.search("." + name)
    return int(match.group(1)) if match else None


def inject_lora(model: nn.Module, targets: list[str], r: int, alpha: float, dropout: float, last_n: int = 0, sides: tuple[str, ...] = ("s0",)) -> list[LoRALinear]:
    """Wrap every linear layer (``nn.Linear`` or GPT-2 ``Conv1D``) whose
    name ends in one of ``targets``.

    ``last_n = 0`` adapts every transformer layer; ``n > 0`` only the last
    ``n`` (a smaller adapter is a cheaper one to synchronise).
    """
    hits = [(name, module) for name, module in model.named_modules() if (isinstance(module, nn.Linear) or _is_conv1d(module)) and name.split(".")[-1] in targets]
    if not hits:
        raise RuntimeError(f"no linear modules matched LoRA targets {targets}")
    indices = [i for i in (_layer_index(name) for name, _ in hits) if i is not None]
    n_layers = max(indices) + 1 if indices else 1
    wrapped = []
    for name, module in hits:
        index = _layer_index(name)
        if last_n > 0 and index is not None and index < n_layers - last_n:
            continue
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        lora = LoRALinear(module, r, alpha, dropout, tuple(sides))
        setattr(parent, child, lora)
        wrapped.append(lora)
    return wrapped
