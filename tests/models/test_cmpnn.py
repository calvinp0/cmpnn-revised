import torch
import pytest

from rdkit import Chem
from cmpnn.models.cmpnn import CMPNNEncoder
from cmpnn.data.molecule_data import MoleculeDataBatch, MoleculeData



def create_dummy_batch(n_atoms=4, n_bonds=6, atom_fdim=133, bond_fdim=14, max_num_bonds=3):
    """Creates a dummy MoleculeDataBatch for encoder testing"""
    f_atoms = torch.randn(n_atoms, atom_fdim)
    f_bonds = torch.randn(n_bonds, bond_fdim)
    a2b = torch.tensor([[1, 2, 0], [3, 0, 0], [4, 5, 0], [0, 0, 0]], dtype=torch.long)
    b2a = torch.tensor([0, 1, 1, 2, 2, 3], dtype=torch.long)
    b2revb = torch.tensor([1, 0, 3, 2, 5, 4], dtype=torch.long)
    a_scope = [(0, n_atoms)]

    return {
        "f_atoms": f_atoms,
        "f_bonds": f_bonds,
        "a2b": a2b,
        "b2a": b2a,
        "b2revb": b2revb,
        "a_scope": a_scope,
    }


@pytest.mark.parametrize("comm_mode", ['add', 'mlp', 'gru', 'ip'])
@pytest.mark.parametrize("booster", ['sum', 'mean', 'sum_max', 'attention'])
def test_cmpnn_encoder_output_shape(comm_mode, booster):
    atom_fdim = 133
    bond_fdim = 14
    hidden_dim = 128
    batch = create_dummy_batch(atom_fdim=atom_fdim, bond_fdim=bond_fdim)

    model = CMPNNEncoder(
        atom_fdim=atom_fdim,
        bond_fdim=bond_fdim,
        atom_messages=True,
        depth=3,
        dropout=0.0,
        hidden_dim=hidden_dim,
        comm_mode=comm_mode,
        booster=booster,
    )
    expected_output = (5, hidden_dim) # 5 atoms in the dummy batch cause we add a dummy
    out = model(**batch)
    assert out.shape == expected_output, f"Output shape mismatch for comm_mode={comm_mode}, booster={booster}"


def test_invalid_comm_mode():
    with pytest.raises(AssertionError):
        CMPNNEncoder(133, 14, atom_messages=True, comm_mode='invalid', booster='sum')


def test_invalid_booster():
    with pytest.raises(AssertionError):
        CMPNNEncoder(133, 14, atom_messages=True, comm_mode='add', booster='invalid')


def test_repr_contains_comm_mode_and_booster():
    model = CMPNNEncoder(133, 14, atom_messages=True, comm_mode='gru', booster='attention')
    repr_str = repr(model)
    assert "comm_mode=gru" in repr_str
    assert "booster=attention" in repr_str

def test_attention_differs_from_sum():
    batch = create_dummy_batch()
    sum_model = CMPNNEncoder(133, 14, atom_messages=True, comm_mode='add', booster='sum')
    attn_model = CMPNNEncoder(133, 14, atom_messages=True, comm_mode='add', booster='attention')
    out_sum = sum_model(**batch)
    out_attn = attn_model(**batch)
    assert not torch.allclose(out_sum, out_attn), "Attention and sum outputs should differ"

