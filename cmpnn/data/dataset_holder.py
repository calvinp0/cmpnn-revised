import numpy as np
import torch
from typing import List, Optional, Sequence


class MoleculeDatasetHolder:
    """Utility to split a :class:`MoleculeDataset` and apply column wise
    transformations to both features and targets.

    Parameters
    ----------
    dataset: MoleculeDataset
        Dataset containing :class:`cmpnn.data.molecule_data.MoleculeData` items.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.feature_transformer = None
        self.target_transformers: Optional[Sequence] = None

    def split(
        self,
        splitter,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        feature_transformer=None,
        target_transformers: Optional[Sequence] = None,
    ):
        """Split the dataset and optionally apply transforms.

        Parameters
        ----------
        splitter: BaseSplitter
            Instance of a splitter implementing ``split`` returning the
            train/val/test ``MoleculeData`` lists.
        train_frac, val_frac, test_frac: float
            Fractions for each split.
        feature_transformer: sklearn-style transformer, optional
            Transformer applied to ``global_features`` of each item.
            Expected to implement ``fit`` and ``transform``.
        target_transformers: sequence of sklearn-style transformers, optional
            One transformer per target column. Each must implement ``fit`` and
            ``transform``.
        """

        train_data, val_data, test_data = splitter.split(
            self.dataset, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac
        )

        self.feature_transformer = feature_transformer
        self.target_transformers = target_transformers

        if self.feature_transformer is not None:
            self._fit_feature_transformer(train_data)
            for split in (train_data, val_data, test_data):
                self._apply_feature_transform(split)

        if self.target_transformers is not None:
            self._fit_target_transformers(train_data)
            for split in (train_data, val_data, test_data):
                self._apply_target_transform(split)

        return train_data, val_data, test_data

    # ------------------------------------------------------------------
    # Internal helpers
    def _collect_features(self, data_list: List) -> np.ndarray:
        feats = [d.global_features.numpy() for d in data_list]
        return np.vstack(feats)

    def _collect_targets(self, data_list: List) -> np.ndarray:
        targets = [d.y.numpy() for d in data_list]
        return np.vstack(targets)

    def _fit_feature_transformer(self, train_data: List) -> None:
        X = self._collect_features(train_data)
        self.feature_transformer.fit(X)

    def _apply_feature_transform(self, data_list: List) -> None:
        for d in data_list:
            arr = d.global_features.numpy().reshape(1, -1)
            transformed = self.feature_transformer.transform(arr)
            d.global_features = torch.tensor(
                transformed.squeeze(), dtype=torch.float32
            )

    def _fit_target_transformers(self, train_data: List) -> None:
        Y = self._collect_targets(train_data)
        for i, transformer in enumerate(self.target_transformers):
            transformer.fit(Y[:, [i]])

    def _apply_target_transform(self, data_list: List) -> None:
        for d in data_list:
            y = d.y.numpy().reshape(1, -1)
            cols = [t.transform(y[:, [i]]) for i, t in enumerate(self.target_transformers)]
            transformed = np.hstack(cols)
            d.y = torch.tensor(transformed.squeeze(), dtype=torch.float32)


class MultiMoleculeDatasetHolder:
    """Same as :class:`MoleculeDatasetHolder` but for
    :class:`cmpnn.featurizer.molecule_dataset.MultiMoleculeDataset` items.
    ``feature_transformers`` must be a sequence with one transformer per
    component in the dataset.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.feature_transformers: Optional[Sequence] = None
        self.target_transformers: Optional[Sequence] = None

    def split(
        self,
        splitter,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        feature_transformers: Optional[Sequence] = None,
        target_transformers: Optional[Sequence] = None,
    ):
        train_data, val_data, test_data = splitter.split(
            self.dataset, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac
        )

        self.feature_transformers = feature_transformers
        self.target_transformers = target_transformers

        if self.feature_transformers is not None:
            for idx, transformer in enumerate(self.feature_transformers):
                X = self._collect_features(train_data, idx)
                transformer.fit(X)
            for split in (train_data, val_data, test_data):
                for idx, _ in enumerate(self.feature_transformers):
                    self._apply_feature_transform(split, idx)

        if self.target_transformers is not None:
            Y = self._collect_targets(train_data)
            for i, transformer in enumerate(self.target_transformers):
                transformer.fit(Y[:, [i]])
            for split in (train_data, val_data, test_data):
                self._apply_target_transform(split)

        return train_data, val_data, test_data

    # Helpers ------------------------------------------------------------
    def _collect_features(self, data_list: List, comp_idx: int) -> np.ndarray:
        feats = [sample[comp_idx].global_features.numpy() for sample in data_list]
        return np.vstack(feats)

    def _apply_feature_transform(self, data_list: List, comp_idx: int) -> None:
        transformer = self.feature_transformers[comp_idx]
        for sample in data_list:
            arr = sample[comp_idx].global_features.numpy().reshape(1, -1)
            transformed = transformer.transform(arr)
            sample[comp_idx].global_features = torch.tensor(
                transformed.squeeze(), dtype=torch.float32
            )

    def _collect_targets(self, data_list: List) -> np.ndarray:
        targets = [sample[0].y.numpy() for sample in data_list]
        return np.vstack(targets)

    def _apply_target_transform(self, data_list: List) -> None:
        for sample in data_list:
            y = sample[0].y.numpy().reshape(1, -1)
            cols = [t.transform(y[:, [i]]) for i, t in enumerate(self.target_transformers)]
            transformed = np.hstack(cols)
            new_tensor = torch.tensor(transformed.squeeze(), dtype=torch.float32)
            for comp in sample:
                comp.y = new_tensor.clone()
