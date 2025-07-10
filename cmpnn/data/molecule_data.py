from typing import List, Optional, Union

import torch
from torch_geometric.data import Data, Batch


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
                 name=None,
                 comp_type=None,
                 idx=None,
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
        self.name = name
        self.comp_type = comp_type
        self.idx = idx

    def __inc__(self, key, value, *args, **kwargs):
        if key in ['a_scope', 'b_scope']:
            return 0  # prevent PyG from incrementing them
        if key in ['a2b', 'bonds']:
            return self.f_atoms.size(0)
        if key in ['b2a']:
            return self.f_bonds.size(0)
        if key in ['input_ids', 'labels']:
            return 0  # ← THIS PREVENTS PAD + INDEX SHIFTING!
        return super().__inc__(key, value, *args, **kwargs)


class MoleculeDataBatch(Batch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def from_data_list(data_list):
        input_ids = []
        labels = []

        batch = MoleculeDataBatch()
        batch.smiles = [mol.smiles for mol in data_list]
        batch.n_mols = len(batch.smiles)
        batch.name = [mol.name for mol in data_list]

        batch.idx = torch.arange(batch.n_mols)


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

        for mol in data_list:
            if hasattr(mol, "input_ids") and mol.input_ids is not None:
                input_ids.append(mol.input_ids)
            if hasattr(mol, "labels") and mol.labels is not None:
                labels.append(mol.labels)

        if input_ids:
            batch.input_ids = torch.cat(input_ids, dim=0)
        if labels:
            batch.labels = torch.cat(labels, dim=0)


        # print(
        #     f"[from_data_list] Init: {t1 - t0:.4f}s | Loop: {t2 - t1:.4f}s | Finalize: {t3 - t2:.4f}s | Total: {t3 - t0:.4f}s")
        return batch


    def to(self, device):
        for name in ['f_atoms', 'f_bonds', 'a2b', 'b2a', 'b2revb', 'bonds', 'global_features', 'y', 'input_ids', 'labels']:
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
        self.names = [comp.name for comp in components]

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

        # 🛠 Patch here: assign idx per pair
        for idx, pair in enumerate(data_list):
            for molecule in pair:
                molecule.idx = torch.tensor(idx)


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
        donor_names = self.names[0] if len(self.names) > 0 else None
        acceptor_names = self.names[1] if len(self.names) > 1 else None
        return (
            f"MultiMoleculeDataBatch("
            f"n_samples={self.n_samples}, "
            f"n_components={self.n_components}, "
            f"smiles={self.smiles}, "
            f"donors={donor_names}, "
            f"acceptors={acceptor_names})"
        )

    @property
    def donor(self):
        return self.components[0]

    @property
    def acceptor(self):
        return self.components[1]

    @property
    def donor_names(self) -> List[str]:
        return self.names[0]

    @property
    def acceptor_names(self) -> List[str]:
        return self.names[1]
