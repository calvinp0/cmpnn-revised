import numpy as np
import torch

from cmpnn.featurizer.utils import featurize_molecule
from cmpnn.featurizer.atom_bond import AtomFeaturizer, BondFeaturizer


def test_extra_atom_features_propagate_to_bonds():
    smiles = "CC"  # ethane has one bond
    af = AtomFeaturizer()
    bf = BondFeaturizer()

    extra = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

    data = featurize_molecule(
        smiles=smiles,
        target=0.0,
        atom_featurizer=af,
        bond_featurizer=bf,
        atom_messages=False,
        extra_atom_features=extra,
    )

    # atoms include extra descriptors
    assert data.f_atoms.shape[1] == len(af) + extra.shape[1]

    # bond feature dimension reflects augmented atom features
    assert data.f_bonds.shape[1] == data.f_atoms.shape[1] + len(bf)

    # bond features begin with the corresponding atom feature vector
    assert torch.allclose(data.f_bonds[0][: data.f_atoms.shape[1]], data.f_atoms[0])
    assert torch.allclose(data.f_bonds[1][: data.f_atoms.shape[1]], data.f_atoms[1])
