import os
import pickle
from collections import Counter
from typing import List
from rdkit import Chem
from tqdm import tqdm

from rmgpy.molecule import Molecule

def atom_to_rmg_vocab(mol: Molecule, includeH: bool = False) -> List[str]:
    """
    Get RMG-style atom type vocab tokens from an RMG molecule.
    """
    atom_tokens = []
    for atom in mol.atoms:
        if atom.is_hydrogen() and not includeH:
            continue
        try:
            atom_tokens.append(atom.atomtype.label)
        except Exception:
            atom_tokens.append('<other>')
    return atom_tokens


def smiles_to_rmg_molecule(smiles: str) -> Molecule:
    """
    Convert SMILES string to RMG molecule.
    """
    try:
        rmg_mol = Molecule(smiles=smiles)
    except Exception as e:
        print(f"Error converting SMILES to RMG molecule: {e} - {smiles}")
        return None

    return rmg_mol


def build_atom_vocab(smiles_list: List[str], min_freq: int = 10, save_path: str = "atom_vocab.pkl"):
    """
    Build a token vocabulary from atom types frmo a list of SMILES strings.
    """
    counter = Counter()
    track_of_smiles = []
    for smi in tqdm(smiles_list, desc="Processing SMILES"):
        mol = smiles_to_rmg_molecule(smi)
        if mol is None:
            continue
        track_of_smiles.append(smi)
        tokens = atom_to_rmg_vocab(mol, includeH=True)
        counter.update(tokens)
    
    # Filter by frequency
    filtered = {k: v for k, v in counter.items() if v >= min_freq}
    
    # Insert special tokens with fixed indices
    special_tokens = ['<pad>', '<other>']
    vocab = {tok: i for i, tok in enumerate(special_tokens)}
    for k in sorted(filtered):
        if k not in vocab:
            vocab[k] = len(vocab)
    # Save the vocab
    with open(save_path, 'wb') as f:
        pickle.dump(vocab, f)

    return vocab, track_of_smiles

def smiles_to_atom_ids(smiles, vocab):
    mol = smiles_to_rmg_molecule(smiles)
    if mol is None:
        return None
    tokens = atom_to_rmg_vocab(mol, includeH=True)
    return [vocab.get(t, vocab['<other>']) for t in tokens]