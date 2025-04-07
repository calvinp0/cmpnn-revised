import torch
import pytest
from cmpnn.data.molecule_data import MoleculeData, MoleculeDataBatch, MultiMoleculeDataBatch


def create_dummy_mol(n_atoms, n_bonds, atom_fdim=133, bond_fdim=14):
    f_atoms = torch.randn(n_atoms, atom_fdim)
    f_bonds = torch.randn(n_bonds, bond_fdim)
    a2b = [[j for j in range(min(n_bonds, 3))] for _ in range(n_atoms)]
    b2a = torch.randint(0, n_atoms, (n_bonds,))
    b2revb = torch.arange(n_bonds - 1, -1, -1)
    a_scope = [(0, n_atoms)]
    b_scope = [(0, n_bonds)]
    y = torch.randn(1)
    return MoleculeData(
        f_atoms=f_atoms,
        f_bonds=f_bonds,
        a2b=a2b,
        b2a=b2a,
        b2revb=b2revb,
        a_scope=a_scope,
        b_scope=b_scope,
        y=y,
        smiles="CCO"
    )


def test_single_batch_structure():
    mols = [create_dummy_mol(5, 6), create_dummy_mol(3, 4)]
    batch = MoleculeDataBatch.from_data_list(mols)

    assert batch.f_atoms.shape[0] == sum(m.f_atoms.shape[0] for m in mols) + 1  # +1 for dummy
    assert batch.a2b.shape[0] == batch.f_atoms.shape[0]
    assert isinstance(batch.a_scope, list) and len(batch.a_scope) == len(mols)
    assert batch.bonds.shape[0] == 2  # shape is [2, N]


def test_multi_component_batch():
    # Give each pair the same target value
    def create_pair(y_val):
        y = torch.tensor([y_val], dtype=torch.float32)
        mol1 = create_dummy_mol(4, 5)
        mol2 = create_dummy_mol(6, 7)
        mol1.y = y
        mol2.y = y
        return [mol1, mol2]

    data = [
        create_pair(1.0),
        create_pair(2.0),
    ]

    multi_batch = MultiMoleculeDataBatch.from_data_list(data)

    assert len(multi_batch) == 2
    assert multi_batch.n_components == 2
    assert multi_batch.components[0].f_atoms.shape[0] > 0
    assert torch.allclose(multi_batch.components[0].y, multi_batch.y)
    assert multi_batch.y.shape == torch.Size([2, 1])


def test_device_transfer():
    mols = [create_dummy_mol(5, 6), create_dummy_mol(3, 4)]
    batch = MoleculeDataBatch.from_data_list(mols)
    batch_cuda = batch.to("cpu")

    assert batch_cuda.f_atoms.device.type == "cpu"
    assert batch_cuda.f_bonds.device.type == "cpu"

    multi_batch = MultiMoleculeDataBatch.from_data_list([[m, m] for m in mols])
    multi_batch = multi_batch.to("cpu")

    for comp in multi_batch.components:
        assert comp.f_atoms.device.type == "cpu"
    if multi_batch.y is not None:
        assert multi_batch.y.device.type == "cpu"

    # Transfer to GPU if available
    if torch.cuda.is_available():
        batch_cuda = batch.to("cuda")
        assert batch_cuda.f_atoms.device.type == "cuda"
        assert batch_cuda.f_bonds.device.type == "cuda"

        multi_batch = MultiMoleculeDataBatch.from_data_list([[m, m] for m in mols])
        multi_batch = multi_batch.to("cuda")

        for comp in multi_batch.components:
            assert comp.f_atoms.device.type == "cuda"
        if multi_batch.y is not None:
            assert multi_batch.y.device.type == "cuda"
