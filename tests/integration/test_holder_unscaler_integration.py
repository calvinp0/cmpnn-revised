import numpy as np
import torch
import pytest
from sklearn.preprocessing import PowerTransformer, StandardScaler

from cmpnn.scaler.unscalers import ColumnUnscaler
from cmpnn.data.dataset_holder import MoleculeDatasetHolder, MultiMoleculeDatasetHolder


class DummyMol:
    def __init__(self, glob_feats, y):
        self.glob_feats = torch.tensor(glob_feats, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)


class EchoSplitter:
    """Returns the same sequence for train/val/test to keep ordering simple"""

    def split(self, dataset, **_):
        """
        Split the dataset into train, validation, and test sets.

        Parameters:
            dataset: The dataset to split.
            **_: Additional keyword arguments.

        Returns:
            A tuple containing the train, validation, and test sets.
        """
        return dataset, list(dataset), list(dataset)


class EchoSplitterMulti:
    """Returns the same sequence for train/val/test to keep ordering simple"""

    def split(self, dataset, **_):
        """
        Split the dataset into train, validation, and test sets.

        Parameters:
            dataset: The dataset to split.
            **_: Additional keyword arguments.

        Returns:
            A tuple containing the train, validation, and test sets.
        """
        return dataset, list(dataset), list(dataset)


# Single Component Holder Test
def test_molecule_holder_with_power_and_standard_inverse_matches_raw():
    rng = np.random.default_rng(0)
    Y_raw = np.hstack(
        [
            rng.normal(size=(64, 1)),  # YJ
            rng.normal(loc=5.0, scale=2, size=(64, 1)),  # StandrdScaler
        ]
    ).astype(np.float64)

    data = [DummyMol(glob_feats=[0.0, 1.0], y=Y_raw[i]) for i in range(len(Y_raw))]

    holder = MoleculeDatasetHolder(data)
    splitter = EchoSplitter()

    tfs = [
        PowerTransformer(method="yeo-johnson", standardize=True),
        StandardScaler(),
    ]

    train, val, test = holder.split(splitter=splitter, target_transformers=tfs)

    Y_scaled = np.stack([np.atleast_1d(d.y.numpy()) for d in train], axis=0)

    un = ColumnUnscaler(holder.target_transformers)
    Y_back = un(torch.tensor(Y_scaled, dtype=torch.float32)).numpy()

    np.testing.assert_allclose(Y_back, Y_raw, rtol=1e-5, atol=1e-6)


# Multi Component Holder Test
def test_multi_holder_target_unscaler_and_component_sync():
    """
    Three target cols: col0 YJ, col1 BC, col2 StdScaler
    """
    rng = np.random.default_rng(42)
    y0 = rng.normal(size=(48, 1))  # YJ
    y1 = rng.lognormal(mean=0.0, sigma=0.6, size=(48, 1))  # Box–Cox (>0)
    y2 = rng.normal(loc=5.0, scale=2.0, size=(48, 1))  # Standard
    Y_raw = np.hstack([y0, y1, y2]).astype(np.float64)

    donor = [DummyMol(glob_feats=[0.0, 1.0], y=Y_raw[i]) for i in range(len(Y_raw))]
    acceptor = [DummyMol(glob_feats=[0.0, 1.0], y=Y_raw[i]) for i in range(len(Y_raw))]
    multi_dataset = [[d, a] for d, a in zip(donor, acceptor)]

    holder = MultiMoleculeDatasetHolder(multi_dataset)
    splitter = EchoSplitterMulti()

    tfs = [
        PowerTransformer(method="yeo-johnson", standardize=True),
        PowerTransformer(method="box-cox", standardize=True),
        StandardScaler(),
    ]

    train, val, test = holder.split(splitter=splitter, target_transformers=tfs)

    # both components should carry the same scaled y
    for sample in train:
        assert torch.allclose(sample[0].y, sample[1].y)

    Y_scaled = np.vstack([sample[0].y.numpy() for sample in train])
    un = ColumnUnscaler(holder.target_transformers)
    # Expect YJ then StdScaler
    assert un.aff_mask.tolist() == [
        False,
        False,
        True,
    ], f"aff_mask={un.aff_mask.tolist()}"
    assert un.pt_mask.tolist() == [True, True, False], f"pt_mask={un.pt_mask.tolist()}"

    Y_back = un(torch.tensor(Y_scaled, dtype=torch.float32)).numpy()

    np.testing.assert_allclose(Y_back, Y_raw, rtol=1e-5, atol=1e-6)
