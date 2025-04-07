import torch
from cmpnn.data.molecule_data import MoleculeData
from cmpnn.data.collate import mol_collate_fn, multimol_collate_fn


def create_dummy_mol(n_atoms=3, n_bonds=4, y_val=1.0):
    atom_fdim = 133
    bond_fdim = 14
    mol = MoleculeData(
        f_atoms=torch.randn(n_atoms, atom_fdim),
        f_bonds=torch.randn(n_bonds, bond_fdim),
        a2b=[[0] * min(n_bonds, 3) for _ in range(n_atoms)],
        b2a=[0] * n_bonds,
        b2revb=[0] * n_bonds,
        a_scope=[(0, n_atoms)],
        b_scope=[(0, n_bonds)],
        y=torch.tensor([y_val]),
        smiles='CCO'
    )
    return mol


def test_mol_collate_fn():
    mols = [create_dummy_mol() for _ in range(4)]
    batch = mol_collate_fn(mols)
    assert batch.f_atoms.shape[0] > 4
    assert batch.y.shape == torch.Size([4, 1])
    assert isinstance(batch, type(mols[0]).__bases__[0])  # Should be MoleculeDataBatch


def test_multimol_collate_fn():
    def make_pair(val):
        y = torch.tensor([val])
        mol1 = create_dummy_mol(y_val=val)
        mol2 = create_dummy_mol(y_val=val)
        mol1.y = y
        mol2.y = y
        return [mol1, mol2]

    pairs = [make_pair(i + 1) for i in range(3)]
    batch = multimol_collate_fn(pairs)

    assert batch.n_samples == 3
    assert batch.n_components == 2
    assert batch.donor.y.shape == torch.Size([3, 1])
    assert torch.allclose(batch.donor.y, batch.y)


def test_empty_collate_fn():
    assert mol_collate_fn([]) is None
    assert multimol_collate_fn([]) is None
