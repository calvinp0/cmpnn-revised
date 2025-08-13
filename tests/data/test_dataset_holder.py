import os
import tempfile
import numpy as np
import pandas as pd
import torch
import pytest

from sklearn.compose import ColumnTransformer
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import PowerTransformer, StandardScaler

from cmpnn.featurizer.atom_bond import AtomFeaturizer, BondFeaturizer
from cmpnn.featurizer.molecule_dataset import MoleculeDataset, MultiMoleculeDataset
from cmpnn.data.dataset_holder import MoleculeDatasetHolder, MultiMoleculeDatasetHolder
from cmpnn.split.random import RandomSplitter
from cmpnn.split.kennard_stone import KennardStoneSplitter


class LengthFeaturizer:
    def __call__(self, smiles: str):
        return np.array([len(smiles)], dtype=float)


def create_csv(single=True):
    if single:
        data = {
            "smiles": ["C", "CC", "CCC", "CCCC", "CCCCC"],
            "t1": [1, 2, 3, 4, 5],
            "t2": [2, 3, 4, 5, 6],
            "t3": [3, 4, 5, 6, 7],
        }
    else:
        data = {
            "smiles1": ["C", "CC", "CCC", "CCCC", "CCCCC"],
            "smiles2": ["O", "N", "F", "Cl", "Br"],
            "t1": [1, 2, 3, 4, 5],
            "t2": [2, 3, 4, 5, 6],
            "t3": [3, 4, 5, 6, 7],
        }
    df = pd.DataFrame(data)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp.name, index=False)
    return tmp.name, df


@pytest.mark.parametrize("splitter_cls", [RandomSplitter, KennardStoneSplitter])
def test_molecule_dataset_holder_transforms(splitter_cls):
    csv_path, df = create_csv(single=True)
    dataset = MoleculeDataset(
        csv_file=csv_path,
        smiles_col="smiles",
        target_cols=["t1", "t2", "t3"],
        atom_featurizer=AtomFeaturizer(),
        bond_featurizer=BondFeaturizer(),
        global_featurizer=LengthFeaturizer(),
        use_cache=False,
    )

    orig_feats = {dataset[i].smiles: dataset[i].global_features.clone() for i in range(len(dataset))}
    orig_targets = {dataset[i].smiles: dataset[i].y.clone() for i in range(len(dataset))}

    splitter = splitter_cls(seed=0)
    feature_tf = ColumnTransformer(
        [
            ("rbf", RBFSampler(gamma=0.5, n_components=2, random_state=0), [0]),
        ]
    )
    target_tfs = [StandardScaler(), StandardScaler(), PowerTransformer(method="yeo-johnson")]

    holder = MoleculeDatasetHolder(dataset)
    train, val, test = holder.split(
        splitter,
        train_frac=0.6,
        val_frac=0.2,
        test_frac=0.2,
        feature_transformer=feature_tf,
        target_transformers=target_tfs,
    )

    # Check shapes
    for split in [train, val, test]:
        for d in split:
            assert d.global_features.shape[0] == 2
            assert d.y.shape[0] == 3

    # Verify transformation correctness
    for split in [train, val, test]:
        for d in split:
            smi = d.smiles
            feat_expected = holder.feature_transformer.transform(orig_feats[smi].numpy().reshape(1, -1)).squeeze()
            np.testing.assert_allclose(d.global_features.numpy(), feat_expected, rtol=1e-6)

            y_orig = orig_targets[smi].numpy().reshape(1, -1)
            cols = [holder.target_transformers[i].transform(y_orig[:, [i]]) for i in range(3)]
            y_expected = np.hstack(cols).squeeze()
            np.testing.assert_allclose(d.y.numpy(), y_expected, rtol=1e-6)

    train_y = np.stack([d.y.numpy() for d in train])
    assert abs(train_y[:, 0].mean()) < 1e-7

    os.remove(csv_path)


@pytest.mark.parametrize("splitter_cls", [RandomSplitter, KennardStoneSplitter])
def test_multi_molecule_dataset_holder_transforms(splitter_cls):
    csv_path, df = create_csv(single=False)
    dataset = MultiMoleculeDataset(
        csv_file=csv_path,
        smiles_cols=["smiles1", "smiles2"],
        target_cols=["t1", "t2", "t3"],
        atom_featurizer=AtomFeaturizer(),
        bond_featurizer=BondFeaturizer(),
        global_featurizer=LengthFeaturizer(),
        use_cache=False,
    )

    orig_feats1 = {}
    orig_feats2 = {}
    orig_targets = {}
    for i in range(len(dataset)):
        c1, c2 = dataset[i]
        key = c1.smiles
        orig_feats1[key] = c1.global_features.clone()
        orig_feats2[key] = c2.global_features.clone()
        orig_targets[key] = c1.y.clone()

    splitter = splitter_cls(seed=0)
    ft1 = ColumnTransformer([
        ("rbf", RBFSampler(gamma=0.5, n_components=2, random_state=0), [0]),
    ])
    ft2 = ColumnTransformer([
        ("rbf", RBFSampler(gamma=0.5, n_components=2, random_state=1), [0]),
    ])
    target_tfs = [StandardScaler(), StandardScaler(), PowerTransformer(method="yeo-johnson")]

    holder = MultiMoleculeDatasetHolder(dataset)
    train, val, test = holder.split(
        splitter,
        train_frac=0.6,
        val_frac=0.2,
        test_frac=0.2,
        feature_transformers=[ft1, ft2],
        target_transformers=target_tfs,
    )

    for split in [train, val, test]:
        for sample in split:
            for comp in sample:
                assert comp.global_features.shape[0] == 2
                assert comp.y.shape[0] == 3

    for split in [train, val, test]:
        for sample in split:
            key = sample[0].smiles
            feat1_exp = holder.feature_transformers[0].transform(orig_feats1[key].numpy().reshape(1, -1)).squeeze()
            feat2_exp = holder.feature_transformers[1].transform(orig_feats2[key].numpy().reshape(1, -1)).squeeze()
            np.testing.assert_allclose(sample[0].global_features.numpy(), feat1_exp, rtol=1e-6)
            np.testing.assert_allclose(sample[1].global_features.numpy(), feat2_exp, rtol=1e-6)

            y_orig = orig_targets[key].numpy().reshape(1, -1)
            cols = [holder.target_transformers[i].transform(y_orig[:, [i]]) for i in range(3)]
            y_expected = np.hstack(cols).squeeze()
            np.testing.assert_allclose(sample[0].y.numpy(), y_expected, rtol=1e-6)
            np.testing.assert_allclose(sample[1].y.numpy(), y_expected, rtol=1e-6)

    os.remove(csv_path)
