import pytest
import torch
from cmpnn.data.molecule_data import MoleculeDataBatch
from cmpnn.featurizer.utils import featurize_molecule
from cmpnn.featurizer.atom_bond import AtomFeaturizer
from cmpnn.featurizer.atom_bond import BondFeaturizer


@pytest.mark.parametrize("smiles", ["C", "N", "[H]", "CC", "CN", "CCl"])
def test_batch_graph_alignment(smiles):
    atom_featurizer = AtomFeaturizer()
    bond_featurizer = BondFeaturizer()

    data = featurize_molecule(smiles, target=0.0,
                               atom_featurizer=atom_featurizer,
                               bond_featurizer=bond_featurizer,
                               atom_messages=False)

    batch = MoleculeDataBatch.from_data_list([data])

    # There should be exactly one molecule in the batch
    assert batch.n_mols == 1
    assert batch.smiles[0] == smiles

    # Check atom slice matches scope
    a_start, a_len = batch.a_scope[0]
    assert a_len == data.f_atoms.size(0)
    torch.testing.assert_close(batch.f_atoms[a_start:a_start+a_len], data.f_atoms)

    # Check bond slice matches scope
    b_start, b_len = batch.b_scope[0]
    assert b_len == data.f_bonds.size(0)
    torch.testing.assert_close(batch.f_bonds[b_start:b_start+b_len], data.f_bonds)

    # Check a2b mapping shape
    assert batch.a2b.shape[0] == batch.f_atoms.size(0)

    # Edge case: If no bonds, f_bonds should still have one dummy
    if data.f_bonds.size(0) == 0:
        assert b_len == 1
        assert torch.allclose(batch.f_bonds[b_start], bond_featurizer(None))
