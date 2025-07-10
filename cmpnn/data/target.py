from dataclasses import dataclass
from typing import Literal, List, Tuple, Optional
import torch
import math

@dataclass(frozen=True)
class TargetSpec:
    name: str # Name of the target
    kind: Literal["cont", "angle", "sincos", "periodic"] # Type of target
    slc: Tuple[int, int]  # (start, end)
    mu   : Optional[float] = None   # ⇤ for z‑scored targets
    std  : Optional[float] = None   # ⇥

class TargetBuilder:
    def __init__(self):
        self.cols  : List[torch.Tensor] = []
        self.specs : List[TargetSpec]   = []
        self.col = 0                    # running pointer

    # ───────── generic append ─────────
    def _append(self, tensor, name, kind, width):
        self.cols.append(tensor)
        self.specs.append(TargetSpec(name, kind,
                                     slice(self.col, self.col + width)))
        self.col += width

    # ───────── concrete helpers ───────
    def add_cont(self, name, value: float):
        self._append(torch.tensor([value], dtype=torch.float32),
                     name, "cont", 1)

    def add_angle_deg(self, name, deg):
        self._append(torch.tensor([deg], dtype=torch.float32),
                     name, "angle", 1)

    def add_sincos_deg(self, name, deg):
        self._append(torch.tensor([deg], dtype=torch.float32),
                     name, "sincos", 1)

    # def add_sincos_deg(self, name, deg):
    #     rad = math.radians(deg)
    #     self._append(torch.tensor([math.sin(rad), math.cos(rad)]),
    #                  name, "sincos", 2)

    def build(self):
        return torch.cat(self.cols), self.specs