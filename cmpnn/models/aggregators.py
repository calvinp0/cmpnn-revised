import torch
import torch.nn as nn
from typing import List, Tuple


class BaseAggregator(nn.Module):
    """
    Base class for aggregating atom-level features into molecule-level representations.
    """

    def forward(self, atom_hiddens: torch.Tensor, a_scope: List[Tuple[int, int]]) -> torch.Tensor:
        raise NotImplementedError("Aggregator must implement forward().")


class MeanAggregator(BaseAggregator):
    """
    Mean aggregation over atom features per molecule.
    """

    def forward(self, atom_hiddens: torch.Tensor, a_scope: List[Tuple[int, int]]) -> torch.Tensor:
        mol_vecs = []
        for a_start, a_size in a_scope:
            if a_size == 0:
                raise ValueError("Molecule with 0 atoms encountered in a_scope.")
            cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
            mol_vecs.append(cur_hiddens.mean(dim=0))
        return torch.stack(mol_vecs, dim=0)


class SumAggregator(BaseAggregator):
    """
    Sum aggregation over atom features per molecule.
    """

    def forward(self, atom_hiddens: torch.Tensor, a_scope: List[Tuple[int, int]]) -> torch.Tensor:
        mol_vecs = []
        for a_start, a_size in a_scope:
            if a_size == 0:
                raise ValueError("Molecule with 0 atoms encountered in a_scope.")
            cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
            mol_vecs.append(cur_hiddens.sum(dim=0))
        return torch.stack(mol_vecs, dim=0)


class NormMeanAggregator(BaseAggregator):
    """
    Mean aggregator with L2 normalization per molecule vector.
    """

    def forward(self, atom_hiddens: torch.Tensor, a_scope: List[Tuple[int, int]]) -> torch.Tensor:
        mol_vecs = []
        for a_start, a_size in a_scope:
            if a_size == 0:
                raise ValueError("Molecule with 0 atoms encountered in a_scope.")
            cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
            mol_vec = cur_hiddens.mean(dim=0)
            mol_vec = mol_vec / (mol_vec.norm(p=2) + 1e-8)  # add epsilon to avoid NaNs
            mol_vecs.append(mol_vec)
        return torch.stack(mol_vecs, dim=0)
