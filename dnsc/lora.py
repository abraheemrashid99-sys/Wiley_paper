"""Minimal multi-adapter LoRA.

One frozen backbone serves both ends of the link. Each wrapped linear layer
holds an independent low-rank adapter per *side* ("tx" for the transmitter,
"rx" for the receiver); the active side is switched before each forward pass.
Keeping the two sides in separate parameter tensors is what lets the receiver
adapters be counted, quantised and synchronised on their own.
"""

from __future__ import annotations

import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

_LAYER_INDEX = re.compile(r"\.(?:layers|h|blocks)\.(\d+)\.")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float, sides: tuple[str, ...]):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        dtype = base.weight.dtype
        for side in sides:
            a = torch.empty(r, base.in_features, dtype=dtype)
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            self.lora_A[side] = nn.Parameter(a)
            self.lora_B[side] = nn.Parameter(torch.zeros(base.out_features, r, dtype=dtype))
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


def inject_lora(
    model: nn.Module,
    targets: list[str],
    r: int,
    alpha: float,
    dropout: float,
    tx_last_n: int = 0,
    rx_last_n: int = 0,
) -> list[LoRALinear]:
    """Wrap every ``nn.Linear`` whose name ends in one of ``targets``.

    ``*_last_n = 0`` adapts every layer for that side; ``n > 0`` restricts the
    side to the last ``n`` transformer layers (a smaller receiver adapter is a
    cheaper one to synchronise).
    """
    hits = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear) and name.split(".")[-1] in targets]
    if not hits:
        raise RuntimeError(f"no nn.Linear modules matched LoRA targets {targets}")
    indices = [i for i in (_layer_index(name) for name, _ in hits) if i is not None]
    n_layers = max(indices) + 1 if indices else 1
    wrapped = []
    for name, module in hits:
        index = _layer_index(name)
        sides = []
        for side, last_n in (("tx", tx_last_n), ("rx", rx_last_n)):
            if last_n <= 0 or index is None or index >= n_layers - last_n:
                sides.append(side)
        if not sides:
            continue
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        lora = LoRALinear(module, r, alpha, dropout, tuple(sides))
        setattr(parent, child, lora)
        wrapped.append(lora)
    return wrapped
