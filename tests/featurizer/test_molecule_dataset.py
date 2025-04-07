import os
import tempfile
import pandas as pd
import torch

from rdkit import Chem
from cmpnn.featurizer.molecule_dataset import MoleculeDataset, MultiMoleculeDataset
from cmpnn.featurizer.atom_bond import AtomFeaturizer, BondFeaturizer
from cmpnn.featurizer.global_feat import ChargeFeaturizer


def create_temp_csv(single=True):
    data = {
        "smiles": ["CCO", "CCC", "C1=CC=CC=C1"],
        "target": [0.1, 0.2, 0.3]
    } if single else {
        "smiles1": ["CCO", "CCC"],
        "smiles2": ["C1=CC=CC=C1", "C=O"],
        "target": [1.0, 0.0]
    }

    df = pd.DataFrame(data)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(temp_file.name, index=False)
    return temp_file.name


def test_molecule_dataset_basic():
    csv_path = create_temp_csv(single=True)
    dataset = MoleculeDataset(
        csv_file=csv_path,
        atom_featurizer=AtomFeaturizer(),
        bond_featurizer=BondFeaturizer(),
        global_featurizer=ChargeFeaturizer(),
        use_cache=False,
    )
    assert len(dataset) == 3
    sample = dataset[0]
    assert hasattr(sample, 'f_atoms')
    assert hasattr(sample, 'f_bonds')
    os.remove(csv_path)


def test_multi_molecule_dataset_basic():
    csv_path = create_temp_csv(single=False)
    dataset = MultiMoleculeDataset(
        csv_file=csv_path,
        atom_featurizer=AtomFeaturizer(),
        bond_featurizer=BondFeaturizer(),
        global_featurizer=ChargeFeaturizer(),
        use_cache=False,
    )
    assert len(dataset) == 2
    sample = dataset[0]
    assert isinstance(sample, list)
    assert all(hasattr(m, 'f_atoms') for m in sample)
    os.remove(csv_path)
