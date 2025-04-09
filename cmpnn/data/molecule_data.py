from typing import List, Optional, Union

from torch_geometric.data import Data, Batch
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence


class MoleculeData(Data):
    def __init__(self,
                 f_atoms=None,
                 f_bonds=None,
                 a2b=None,
                 b2a=None,
                 a_scope=None,
                 b_scope=None,
                 global_features=None,
                 y=None,
                 bonds=None,
                 smiles=None,
                 b2revb=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.f_atoms = f_atoms
        self.f_bonds = f_bonds
        self.a2b = a2b
        self.b2a = b2a
        self.a_scope = a_scope
        self.b_scope = b_scope
        self.global_features = global_features
        self.y = y
        self.bonds = bonds
        self.smiles = smiles
        self.b2revb = b2revb

    def __inc__(self, key, value, *args, **kwargs):
        if key in ['a_scope', 'b_scope']:
            return 0  # prevent PyG from incrementing them
        if key in ['a2b', 'bonds']:
            return self.f_atoms.size(0)
        if key in ['b2a']:
            return self.f_bonds.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class MoleculeDataBatch(Batch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def from_data_list(data_list):
        import time
        t0 = time.perf_counter()

        batch = MoleculeDataBatch()
        batch.smiles = [mol.smiles for mol in data_list]
        batch.n_mols = len(batch.smiles)

        atom_fdim = data_list[0].f_atoms.size(1)
        bond_fdim = data_list[0].f_bonds.size(1)

        n_atoms = 1
        n_bonds = 1
        a_scope = []
        b_scope = []

        f_atoms = [torch.zeros(1, atom_fdim)]
        f_bonds = [torch.zeros(1, bond_fdim)]
        a2b = [[]]
        b2a = [0]
        b2revb = [0]
        bonds = [[0, 0]]

        t1 = time.perf_counter()

        for mol in data_list:
            f_atoms.append(mol.f_atoms)
            f_bonds.append(mol.f_bonds)

            for a in range(mol.f_atoms.size(0)):
                a2b.append([b + n_bonds for b in mol.a2b[a]])

            # for b in range(mol.f_bonds.size(0)):
            #     b2a.append(n_atoms + mol.b2a[b])
            #     b2revb.append(n_bonds + mol.b2revb[b])
            #     bonds.append([n_atoms + mol.b2a[b], n_atoms + mol.b2a[mol.b2revb[b]]])
            if len(mol.b2a) == mol.f_bonds.size(0):
                for b in range(mol.f_bonds.size(0)):
                    b2a.append(n_atoms + mol.b2a[b])
                    b2revb.append(n_bonds + mol.b2revb[b])
                    bonds.append([
                        n_atoms + mol.b2a[b],
                        n_atoms + mol.b2a[mol.b2revb[b]]
                    ])
            else:
                # Handle 1-atom molecule with dummy bond
                b2a.append(0)
                b2revb.append(0)
                bonds.append([0, 0])

            a_scope.append((n_atoms, mol.f_atoms.size(0)))
            b_scope.append((n_bonds, mol.f_bonds.size(0)))

            n_atoms += mol.f_atoms.size(0)
            n_bonds += mol.f_bonds.size(0)

        t2 = time.perf_counter()

        batch.max_num_bonds = max(1, max(len(x) for x in a2b))
        padded_a2b = [
            x[:batch.max_num_bonds] + [0] * (batch.max_num_bonds - len(x)) for x in a2b
        ]

        bonds = torch.tensor(bonds, dtype=torch.long).T if bonds else torch.zeros((2, 0), dtype=torch.long)

        batch.f_atoms = torch.cat(f_atoms, dim=0)
        batch.f_bonds = torch.cat(f_bonds, dim=0)
        batch.a2b = torch.tensor(padded_a2b, dtype=torch.long)
        batch.b2a = torch.tensor(b2a, dtype=torch.long)
        batch.b2revb = torch.tensor(b2revb, dtype=torch.long)
        batch.bonds = bonds
        batch.a_scope = a_scope
        batch.b_scope = b_scope

        if hasattr(data_list[0], "global_features") and data_list[0].global_features is not None:
            batch.global_features = torch.stack([mol.global_features for mol in data_list])
        if hasattr(data_list[0], "y") and data_list[0].y is not None:
            batch.y = torch.stack([mol.y if mol.y.dim() > 0 else mol.y.unsqueeze(0) for mol in data_list])

        t3 = time.perf_counter()

        # print(
        #     f"[from_data_list] Init: {t1 - t0:.4f}s | Loop: {t2 - t1:.4f}s | Finalize: {t3 - t2:.4f}s | Total: {t3 - t0:.4f}s")
        return batch

    # @staticmethod
    # def from_data_list(data_list):
    #     batch = MoleculeDataBatch()
    #     batch.smiles = [mol.smiles for mol in data_list]
    #     batch.n_mols = len(batch.smiles)

    #     # Atom/bond dimensions
    #     atom_fdim = data_list[0].f_atoms.size(1)
    #     bond_fdim = data_list[0].f_bonds.size(1)

    #     # Totals
    #     n_atoms_total = sum(mol.f_atoms.size(0) for mol in data_list)
    #     n_bonds_total = sum(mol.f_bonds.size(0) for mol in data_list)

    #     # Preallocate
    #     f_atoms = torch.zeros((n_atoms_total + 1, atom_fdim))
    #     f_bonds = torch.zeros((n_bonds_total + 1, bond_fdim))
    #     b2a = torch.zeros(n_bonds_total + 1, dtype=torch.long)
    #     b2revb = torch.zeros(n_bonds_total + 1, dtype=torch.long)
    #     bonds = torch.zeros((n_bonds_total + 1, 2), dtype=torch.long)

    #     global_features = []
    #     targets = []

    #     # For a2b: need to pad to max num bonds
    #     max_num_bonds = max(len(bonds) for mol in data_list for bonds in mol.a2b)
    #     a2b = torch.zeros((n_atoms_total + 1, max_num_bonds), dtype=torch.long)

    #     a_scope = []
    #     b_scope = []

    #     atom_offset = 1
    #     bond_offset = 1

    #     for mol in data_list:
    #         na = mol.f_atoms.size(0)
    #         nb = mol.f_bonds.size(0)

    #         f_atoms[atom_offset:atom_offset + na] = mol.f_atoms
    #         f_bonds[bond_offset:bond_offset + nb] = mol.f_bonds

    #         for a in range(na):
    #             bond_ids = mol.a2b[a]
    #             a2b[atom_offset + a, :len(bond_ids)] = torch.tensor(
    #                 [bond_offset + b for b in bond_ids], dtype=torch.long
    #             )

    #         mol_b2a = torch.tensor(mol.b2a, dtype=torch.long)
    #         mol_b2revb = mol.b2revb.clone().detach().to(dtype=torch.long)

    #         b2revb[bond_offset:bond_offset + nb] = mol.b2revb.clone().detach() + bond_offset
    #         b2revb[bond_offset:bond_offset + nb] = mol.b2revb.clone().detach() + bond_offset

    #         for i in range(nb):
    #             b2 = mol_b2a[i].item()
    #             rb2 = mol_b2revb[i].item()
    #             bonds[bond_offset + i, 0] = atom_offset + b2
    #             bonds[bond_offset + i, 1] = atom_offset + mol_b2a[rb2].item()

    #         a_scope.append((atom_offset, na))
    #         b_scope.append((bond_offset, nb))

    #         if hasattr(mol, "global_features") and mol.global_features is not None:
    #             global_features.append(mol.global_features)

    #         if hasattr(mol, "y") and mol.y is not None:
    #             y = mol.y
    #             if y.dim() == 0:
    #                 y = y.unsqueeze(0)
    #             targets.append(y)

    #         atom_offset += na
    #         bond_offset += nb

    #     batch.f_atoms = f_atoms
    #     batch.f_bonds = f_bonds
    #     batch.a2b = a2b
    #     batch.b2a = b2a
    #     batch.b2revb = b2revb
    #     batch.bonds = bonds.T  # shape [2, N] if required by model
    #     batch.a_scope = a_scope
    #     batch.b_scope = b_scope

    #     if global_features:
    #         batch.global_features = torch.stack(global_features)

    #     if targets:
    #         batch.y = torch.stack(targets)

    #     return batch

    # @staticmethod
    # def from_data_list(data_list):
    #     batch = MoleculeDataBatch()
    #     batch.smiles = [mol.smiles for mol in data_list]
    #     batch.n_mols = len(batch.smiles)

    #     atom_fdim = data_list[0].f_atoms.size(1)
    #     bond_fdim = data_list[0].f_bonds.size(1)

    #     batch.n_atoms = 1
    #     batch.n_bonds = 1
    #     batch.a_scope = []
    #     batch.b_scope = []

    #     f_atoms = [torch.zeros(1, atom_fdim)]  # dummy row to align indexing
    #     f_bonds = [torch.zeros(1, bond_fdim)]  # dummy row for bond features
    #     a2b = [[]]
    #     b2a = [0]
    #     b2revb = [0]
    #     bonds = [[0, 0]]

    #     for data in data_list:
    #         f_atoms.append(data.f_atoms)
    #         # f_bonds.extend(data.f_bonds)
    #         f_bonds.append(data.f_bonds)

    #         for a in range(data.f_atoms.shape[0]):
    #             a2b.append([b + batch.n_bonds for b in data.a2b[a]])

    #         for b in range(data.f_bonds.shape[0]):
    #             b2a_idx = batch.n_atoms + data.b2a[b]
    #             rev_b_idx = data.b2revb[b]
    #             b2a.append(b2a_idx)
    #             b2revb.append(batch.n_bonds + rev_b_idx)
    #             bonds.append([
    #                 b2a_idx,
    #                 batch.n_atoms + data.b2a[rev_b_idx]
    #             ])
    #         batch.a_scope.append((batch.n_atoms, data.f_atoms.shape[0]))
    #         batch.b_scope.append((batch.n_bonds, data.f_bonds.shape[0]))

    #         batch.n_atoms += data.f_atoms.shape[0]
    #         batch.n_bonds += data.f_bonds.shape[0]
    #     # batch.max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b))
    #     # # Pad a2b so that it has shape [n_atoms, max_num_bonds]
    #     # padded_a2b = [
    #     #     bonds[:batch.max_num_bonds] + [0] * (batch.max_num_bonds - len(bonds))
    #     #     for bonds in a2b
    #     # ]

    #     ### NEW ###
    #     a2b_tensors = [torch.tensor(in_bonds, dtype=torch.long) for in_bonds in a2b]
    #     padded_a2b = pad_sequence(a2b_tensors, batch_first=True, padding_value=0)
    #     batch.max_num_bonds = padded_a2b.shape[1]
    #     ####

    #     if len(bonds) == 0:
    #         bonds = torch.zeros((0, 2), dtype=torch.long)
    #     else:
    #         bonds = torch.LongTensor(bonds)

    #     # Convert everything to tensors
    #     batch.f_atoms = torch.cat(f_atoms, dim=0)
    #     # batch.f_bonds = torch.FloatTensor(f_bonds)
    #     batch.f_bonds = torch.cat(f_bonds, dim=0)
    #     batch.a2b = torch.LongTensor(padded_a2b)
    #     batch.b2a = torch.LongTensor(b2a)
    #     batch.b2revb = torch.LongTensor(b2revb)
    #     batch.bonds = torch.LongTensor(bonds)
    #     batch.a2a = None
    #     batch.b2b = None

    #     # Optional: global features or target aggregation
    #     if hasattr(data_list[0], "global_features") and data_list[0].global_features is not None:
    #         batch.global_features = torch.stack([mol.global_features for mol in data_list])
    #     if hasattr(data_list[0], "y") and data_list[0].y is not None:
    #         batch.y = torch.stack([mol.y if mol.y.dim() > 0 else mol.y.unsqueeze(0) for mol in data_list])

    #     return batch

    def to(self, device):
        for name in ['f_atoms', 'f_bonds', 'a2b', 'b2a', 'b2revb', 'bonds', 'global_features', 'y']:
            val = getattr(self, name, None)
            if torch.is_tensor(val):
                setattr(self, name, val.to(device, non_blocking=True))
        return self


class MultiMoleculeDataBatch:
    def __init__(self, components: List[MoleculeDataBatch], y: Optional[torch.Tensor] = None):
        self.components = components
        self.n_components = len(components)
        self.n_samples = components[0].n_mols

        for comp in components:
            assert comp.n_mols == self.n_samples, "All components must have the same number of samples"

        self.smiles = [comp.smiles for comp in components]

        # Ensure y is shared or consistent across components
        if y is not None:
            self.y = y
        else:
            y_vals = [getattr(comp, 'y', None) for comp in components]
            if all(y is not None for y in y_vals):
                base_y = y_vals[0]
                assert all(torch.allclose(base_y, y_i) for y_i in y_vals[1:]), "Mismatched y values across components"
                self.y = base_y
            else:
                self.y = None  # Acceptable case for inference
                print("Warning: y values are missing or inconsistent across components.")

    @staticmethod
    def from_data_list(data_list: List[List['MoleculeData']]) -> 'MultiMoleculeDataBatch':
        """
        Create a MultiMoleculeDataBatch from a list of molecule pairs (or tuples),
        e.g., [[mol1, mol2], [mol3, mol4], ...]
        """
        if not all(len(pair) == len(data_list[0]) for pair in data_list):
            raise ValueError("All samples must have the same number of molecular components.")

        components = list(zip(*data_list))
        batched_components = [MoleculeDataBatch.from_data_list(list(group)) for group in components]

        # Optionally infer shared y from components
        y_vals = [getattr(comp, 'y', None) for comp in batched_components]
        if all(y is not None for y in y_vals):
            base_y = y_vals[0]
            assert all(torch.allclose(base_y, y_i) for y_i in y_vals[1:]), "Mismatched y values across components"
            return MultiMoleculeDataBatch(batched_components, y=base_y)
        else:
            return MultiMoleculeDataBatch(batched_components)

    def __getitem__(self, idx: int) -> List:
        return [comp[idx] for comp in self.components]

    def __len__(self) -> int:
        return self.n_samples

    def to(self, device: Union[str, torch.device]) -> 'MultiMoleculeDataBatch':
        return MultiMoleculeDataBatch([comp.to(device) for comp in self.components],
                                      y=self.y.to(device) if self.y is not None else None)

    def __repr__(self):
        return f"MultiMoleculeDataBatch(n_samples={self.n_samples}, n_components={self.n_components})"

    @property
    def donor(self):
        return self.components[0]

    @property
    def acceptor(self):
        return self.components[1]
