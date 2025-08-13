import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from cmpnn.data.molecule_data import MoleculeData
from typing import List, Callable, Optional
import pickle

def featurize_molecule(smiles: str, target, atom_featurizer, bond_featurizer, global_featurizer=None,
                       atom_messages=False, extra_atom_features=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    n_atoms = 0
    n_bonds = 0
    f_atoms = []
    f_bonds = []
    a2b = []
    b2a = []
    b2revb = []
    bonds = []
    a_scope = []
    b_scope = []

    n_atoms = mol.GetNumAtoms()

    for i, atom in enumerate(mol.GetAtoms()):
        f_atoms.append(atom_featurizer(atom))
    f_atoms = [f_atoms[i] for i in range(n_atoms)]

    for _ in range(n_atoms):
        a2b.append([])

    # Get Bond Features
    for atom1 in range(n_atoms):
        for atom2 in range(atom1 + 1, n_atoms):
            bond = mol.GetBondBetweenAtoms(atom1, atom2)
            if bond is None:
                continue

            f_bond = bond_featurizer(bond)

            if atom_messages:
                f_bonds.append(f_bond)
                f_bonds.append(f_bond)
            else:
                f_bonds.append(torch.cat([f_atoms[atom1], f_bond]))
                f_bonds.append(torch.cat([f_atoms[atom2], f_bond]))

            # Update index mappings
            b1 = n_bonds
            b2 = n_bonds + 1
            a2b[atom2].append(b1)  # b1 = a1 --> a2
            b2a.append(atom1)
            a2b[atom1].append(b2)  # b2 = a2 --> a1
            b2a.append(atom2)
            b2revb.append(b2)
            b2revb.append(b1)
            n_bonds += 2
            bonds.append(np.array([atom1, atom2]))

    # Convert features to tensors
    f_atoms = torch.stack(f_atoms)

    if extra_atom_features is not None:
        extra_atom_features = torch.tensor(extra_atom_features, dtype=torch.float32)
        if extra_atom_features.dim() == 1:
            extra_atom_features = extra_atom_features.unsqueeze(0)
        if extra_atom_features.size(0) != f_atoms.size(0):
            raise ValueError("Mismatch between number of atoms and extra descriptors")
        f_atoms = torch.cat([f_atoms, extra_atom_features], dim=1)

    if len(f_bonds) == 0:
        if atom_messages:
            f_bonds.append(bond_featurizer(None))
        else:
            f_bonds.append(torch.cat([f_atoms[0], bond_featurizer(None)]))
            
    f_bonds = torch.stack(f_bonds)


    a2b = a2b
    b2a = b2a
    b2revb = torch.tensor(b2revb, dtype=torch.long)
    bonds = torch.tensor(np.array(bonds), dtype=torch.long)
    a_scope = [(0, n_atoms)]
    b_scope = [(0, n_bonds)]
    global_features = None
    if global_featurizer is not None:
        global_features = torch.tensor(global_featurizer(smiles), dtype=torch.float)

    # Create MoleculeData object
    return MoleculeData(
        f_atoms=f_atoms,
        f_bonds=f_bonds,
        a2b=a2b,
        b2a=b2a,
        a_scope=a_scope,
        b_scope=b_scope,
        global_features=global_features,
        y=torch.tensor([target], dtype=torch.float),
        bonds=bonds,
        smiles=smiles,
        b2revb=b2revb,
    )


def infer_dtype(series: pd.Series) -> torch.dtype:
    if pd.api.types.is_integer_dtype(series):
        return torch.long
    elif pd.api.types.is_float_dtype(series):
        return torch.float32
    elif pd.api.types.is_bool_dtype(series):
        return torch.float32
    elif pd.api.types.is_object_dtype(series):
        return torch.long 
    else:
        raise ValueError(f"Unsupported target type: {series.dtype}")
