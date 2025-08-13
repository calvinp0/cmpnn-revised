import numpy as np

from cmpnn.split.kennard_stone import KennardStoneSplitter, kennard_stone_order, _featurize_smiles


class SimpleMol:
    def __init__(self, smiles):
        self.smiles = smiles


def test_kennard_stone_single_split():
    smiles = ["C", "CC", "CCC", "CCCC", "CCCCC"]
    dataset = [SimpleMol(s) for s in smiles]
    splitter = KennardStoneSplitter(seed=0)

    train, val, test = splitter.split(dataset, train_frac=0.6, val_frac=0.2, test_frac=0.2)
    assert len(train) == 3
    assert len(val) == 1
    assert len(test) == 1

    # determinism
    train2, val2, test2 = splitter.split(dataset, train_frac=0.6, val_frac=0.2, test_frac=0.2)
    assert [m.smiles for m in train] == [m.smiles for m in train2]
    assert [m.smiles for m in val] == [m.smiles for m in val2]
    assert [m.smiles for m in test] == [m.smiles for m in test2]

    # verify order matches internal KS algorithm
    X = _featurize_smiles(smiles, splitter.radius, splitter.n_bits)
    D = splitter._build_D(X)
    expected_order = kennard_stone_order(D, seed=0)
    all_smiles = [m.smiles for m in dataset]
    ordered_smiles = [all_smiles[i] for i in expected_order]
    assert [m.smiles for m in train + val + test] == ordered_smiles


def test_kennard_stone_pair_split_return_indices():
    donors = ["C", "CC", "CCC", "CCCC", "CCCCC"]
    acceptors = ["O", "N", "F", "Cl", "Br"]
    dataset = [(SimpleMol(d), SimpleMol(a)) for d, a in zip(donors, acceptors)]
    splitter = KennardStoneSplitter(seed=0, joint_mode="mean")

    train_idx, val_idx, test_idx = splitter.split(
        dataset, train_frac=0.6, val_frac=0.2, test_frac=0.2, return_indices=True
    )
    assert len(train_idx) == 3
    assert len(val_idx) == 1
    assert len(test_idx) == 1

    # ensure indices are unique and cover all samples
    all_idx = sorted(train_idx + val_idx + test_idx)
    assert all_idx == list(range(len(dataset)))

    # determinism
    train_idx2, val_idx2, test_idx2 = splitter.split(
        dataset, train_frac=0.6, val_frac=0.2, test_frac=0.2, return_indices=True
    )
    assert train_idx == train_idx2
    assert val_idx == val_idx2
    assert test_idx == test_idx2

