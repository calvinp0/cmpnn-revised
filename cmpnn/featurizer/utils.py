import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from cmpnn.data.molecule_data import MoleculeData


def featurize_molecule(mol_or_smiles, target, atom_featurizer, bond_featurizer, global_featurizer=None,
                       atom_messages=False, add_hs=False, sanitize=True, name=None, comp_type=None, mol_properties= None, electro_map=None, idx=None):
    if isinstance(mol_or_smiles, Chem.Mol):
        mol = mol_or_smiles
        if sanitize:
            Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol)
        # Note: if sanitize is False, we assume the molecule is already in the desired state.
    elif isinstance(mol_or_smiles, str):
        mol = Chem.MolFromSmiles(mol_or_smiles, sanitize=sanitize)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {mol_or_smiles}")
        if not sanitize:
            # Perform custom sanitization without altering properties
            Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            mol.UpdatePropertyCache(strict=False)
    else:
        raise TypeError("Input must be either an RDKit Mol object or a SMILES string.")

    # Optionally add explicit hydrogens
    if add_hs:
        mol = Chem.AddHs(mol)

    # Get the atom properties

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
        base_feat  = atom_featurizer(atom)
        label_feat = encode_labels(i, mol_properties)
        if electro_map is not None:
            # pull electro_map entry or empty dict
            e = (electro_map or {}).get(str(i), {})
            # raw values or zero
            R_val = float(e.get("R")) if e.get("R") is not None else 0.0
            A_val = float(e.get("A")) if e.get("A") is not None else 0.0
            D_val = float(e.get("D")) if e.get("D") is not None else 0.0

            R_mask = 1.0 if e.get("R") is not None else 0.0
            A_mask = 1.0 if e.get("A") is not None else 0.0
            D_mask = 1.0 if e.get("D") is not None else 0.0

            # Apply transforms:
            R_feat = torch.tensor([R_val])
            A_feat = torch.tensor([A_val])
            D_feat = torch.tensor([
                np.sin(np.deg2rad(D_val)),
                np.cos(np.deg2rad(D_val)),
            ], dtype=torch.float32)

            mask_feat = torch.tensor([R_mask, A_mask, D_mask], dtype=torch.float32)

            electro_feats = torch.cat([R_feat, A_feat, D_feat, mask_feat])
            f_atoms.append(torch.cat([base_feat, label_feat, electro_feats]))
        else:
            f_atoms.append(torch.cat([base_feat, label_feat]))
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

    if target is None:
        y = None
    elif isinstance(target, torch.Tensor):
        # If target is already a tensor, ensure it's float type.
        y = target.float()
    else:
        # Assume target is a sequence and convert it directly.
        y = torch.tensor(target, dtype=torch.float)

    # Create MoleculeData object
    return MoleculeData(
        f_atoms=f_atoms,
        f_bonds=f_bonds,
        a2b=a2b,
        b2a=b2a,
        a_scope=a_scope,
        b_scope=b_scope,
        global_features=global_features,
        y=y,
        bonds=bonds,
        smiles=smiles,
        b2revb=b2revb,
        name=name,
        comp_type=comp_type,
        idx=idx

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


LABEL_VOCAB = ["acceptor", "a_hydrogen", "donator", "d_hydrogen"]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABEL_VOCAB)}

def encode_labels(atom_idx, mol_properties, num_labels=len(LABEL_VOCAB)):
    """
    Returns a one-hot label vector of shape (num_labels,) for the given atom index.
    """
    one_hot = torch.zeros(num_labels)
    if str(atom_idx) in mol_properties:
        label = mol_properties[str(atom_idx)].get("label")
        if label in LABEL_TO_INDEX:
            one_hot[LABEL_TO_INDEX[label]] = 1.0
    return one_hot