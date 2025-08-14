import numpy as np
import torch
from typing import List, Optional, Sequence
from cmpnn.scaler.unscalers import ColumnUnscaler


def _ensure_raw_single(obj) -> None:
    if hasattr(obj, "y") and obj.y is not None and not hasattr(obj, "y_raw"):
        obj.y_raw = obj.y.detach().clone()
    if (
        hasattr(obj, "global_features")
        and obj.global_features is not None
        and not hasattr(obj, "global_features_raw")
    ):
        obj.global_features_raw = obj.global_features.detach().clone()
    if (
        hasattr(obj, "f_atoms")
        and obj.f_atoms is not None
        and not hasattr(obj, "f_atoms_raw")
    ):
        obj.f_atoms_raw = obj.f_atoms.detach().clone()
    if (
        hasattr(obj, "f_bonds")
        and obj.f_bonds is not None
        and not hasattr(obj, "f_bonds_raw")
    ):
        obj.f_bonds_raw = obj.f_bonds.detach().clone()


def _ensure_raw_list(objs: List) -> None:
    for o in objs:
        _ensure_raw_single(o)


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
        self.n_targets_: Optional[int] = None
        self.extra_atom_start: int = 0
        self.mirror_into_bonds: Optional[bool] = None  # None = auto-detect

    def split(
        self,
        splitter,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        feature_transformer=None,
        target_transformers: Optional[Sequence] = None,
        atom_desc_scaler=None,
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
        _ensure_raw_list(train_data)
        _ensure_raw_list(val_data)
        _ensure_raw_list(test_data)

        self.feature_transformer = feature_transformer
        self.target_transformers = target_transformers
        self.atom_desc_scaler = atom_desc_scaler
        self.n_targets_ = (
            len(target_transformers) if target_transformers is not None else None
        )  # <-- add this

        if self.feature_transformer is not None:
            self._fit_feature_transformer(train_data)
            for split in (train_data, val_data, test_data):
                self._apply_feature_transform(split)

        if self.target_transformers is not None:
            self._fit_target_transformers(train_data)
            for split in (train_data, val_data, test_data):
                self._apply_target_transform(split)

        if self.atom_desc_scaler is not None:
            self.atom_desc_scaler.fit(train_data)
            for split in (train_data, val_data, test_data):
                self.atom_desc_scaler.transform(split)
        return train_data, val_data, test_data

    # ------------------------------------------------------------------
    # Internal helpers

    def _auto_detect_mirror(self, d) -> bool:
        A = getattr(d, "f_atoms_raw", getattr(d, "f_atoms", None))
        B = getattr(d, "f_bonds", None)  # use current; raw not needed for layout check
        b2a = getattr(d, "b2a", None)
        if A is None or B is None or b2a is None:
            return False
        atom_dim = A.shape[1]
        if B.shape[1] < atom_dim:
            return False
        src = torch.as_tensor(b2a, dtype=torch.long, device=A.device)

        return torch.allclose(B[:, :atom_dim], A.index_select(0, src))

    def _collect_features(self, data_list: List) -> np.ndarray:
        feats = [
            d.global_features_raw.numpy()
            for d in data_list
            if hasattr(d, "global_features_raw")
        ]
        return np.vstack(feats) if len(feats) else np.zeros((0,))

    def _collect_extra_atom_column(self, data_list: List, col_idx: int) -> np.ndarray:
        """Stack one extra column across ALL atoms of ALL molecules -> (N_total_atoms, 1)."""
        rows = []
        for d in data_list:
            A = getattr(d, "f_atoms_raw", None)
            if A is not None:
                col = (
                    A[:, [self.extra_atom_start + col_idx]].cpu().numpy()
                )  # (n_atoms, 1)
                if col.size:
                    rows.append(col)
        return np.vstack(rows) if rows else np.zeros((0, 1), dtype=np.float32)

    def _collect_targets(self, data_list: List) -> np.ndarray:
        targets = [d.y_raw.numpy() for d in data_list if hasattr(d, "y_raw")]
        return np.vstack(targets) if len(targets) else np.zeros((0,))

    def _fit_feature_transformer(self, train_data: List) -> None:
        X = self._collect_features(train_data)
        if X.size:
            self.feature_transformer.fit(X)

    def _fit_extra_atom_transformers(self, train_data: List) -> None:
        # Decide mirroring if user did not specify
        if self.mirror_into_bonds is None:
            self.mirror_into_bonds = False
            for d in train_data:
                if self._auto_detect_mirror(d):
                    self.mirror_into_bonds = True
                    break
        # Fit each per-column transformer
        for i, tf in enumerate(self.extra_atom_transformers):
            X = self._collect_extra_atom_column(train_data, i)  # (N_total_atoms, 1)
            if X.size:
                tf.fit(X)

    @torch.no_grad()
    def _apply_extra_atom_transformers(self, data_list: List) -> None:
        atom_start = self.extra_atom_start
        do_mirror = bool(self.mirror_into_bonds)
        for d in data_list:
            Araw = getattr(d, "f_atoms_raw", None)
            if Araw is None:
                continue
            A = Araw.clone()
            # transform each extra column independently
            for i, tf in enumerate(self.extra_atom_transformers):
                col = Araw[:, [atom_start + i]].cpu().numpy()  # (n_atoms, 1)
                col_t = tf.transform(col)  # (n_atoms, 1)
                A[:, atom_start + i] = torch.from_numpy(col_t.squeeze(1)).to(A.dtype)
            d.f_atoms = A

            if do_mirror:
                B = getattr(d, "f_bonds", None)
                b2a = getattr(d, "b2a", None)
                if B is not None and b2a is not None:
                    atom_dim = d.f_atoms.shape[1]
                    if B.shape[1] >= atom_dim:
                        src = torch.as_tensor(b2a, dtype=torch.long, device=A.device)
                        B[:, :atom_dim] = d.f_atoms.index_select(0, src)

    def _apply_feature_transform(self, data_list: List) -> None:
        if self.feature_transformer is None:
            return
        for d in data_list:
            if not hasattr(d, "global_features_raw") or d.global_features_raw is None:
                continue
            arr = d.global_features_raw.numpy().reshape(1, -1)
            transformed = self.feature_transformer.transform(arr)
            d.global_features = torch.tensor(transformed.squeeze(), dtype=torch.float32)

    def _fit_target_transformers(self, train_data: List) -> None:
        Y = self._collect_targets(train_data)
        if self.n_targets_ is None and Y.size:
            self.n_targets_ = Y.shape[1]
        if self.target_transformers is not None and Y.size:
            assert self.n_targets_ == len(self.target_transformers), (
                f"len(target_transformers)={len(self.target_transformers)} "
                f"but n_targets_={self.n_targets_}"
            )
            assert (
                Y.shape[1] == self.n_targets_
            ), f"Training targets have {Y.shape[1]} columns, expected {self.n_targets_}"
            # Optional: Box–Cox domain guard
            for i, t in enumerate(self.target_transformers):
                if (
                    t.__class__.__name__ == "PowerTransformer"
                    and getattr(t, "method", "") == "box-cox"
                ):
                    if not np.all(Y[:, i] > 0):
                        bad = np.where(Y[:, i] <= 0)[0][:5]
                        raise ValueError(
                            f"Box–Cox column {i} has non-positive values at indices {bad.tolist()}"
                        )
            for i, transformer in enumerate(self.target_transformers):
                transformer.fit(Y[:, [i]])

    def _apply_target_transform(self, data_list: List) -> None:
        if self.target_transformers is None:
            return
        for d in data_list:
            if not hasattr(d, "y_raw") or d.y_raw is None:
                continue
            y = d.y_raw.numpy().reshape(1, -1)
            assert self.n_targets_ is not None, "n_targets_ was not initialized"
            assert (
                y.shape[1] == self.n_targets_
            ), f"y has {y.shape[1]} cols but {self.n_targets_} transformers were provided"
            cols = [
                t.transform(y[:, [i]]) for i, t in enumerate(self.target_transformers)
            ]
            transformed = np.hstack(cols).reshape(-1)
            d.y = torch.tensor(transformed, dtype=torch.float32)

    # ---- convenience ----
    def build_target_unscaler(self):
        return (
            ColumnUnscaler(self.target_transformers)
            if self.target_transformers
            else None
        )

    def reset_to_raw(self, data_list: Optional[List] = None) -> None:
        """Restore y/global_features to their raw snapshots."""
        if data_list is None:
            data_list = self.dataset
        for d in data_list:
            if hasattr(d, "y_raw"):
                d.y = d.y_raw.detach().clone()
            if hasattr(d, "global_features_raw"):
                d.global_features = d.global_features_raw.detach().clone()
            if hasattr(d, "f_atoms_raw"):
                d.f_atoms = d.f_atoms_raw.detach().clone()
            if hasattr(d, "f_bonds_raw"):
                d.f_bonds = d.f_bonds_raw.detach().clone()


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
        self.n_targets_: Optional[int] = None

    def split(
        self,
        splitter,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        feature_transformers: Optional[Sequence] = None,
        target_transformers: Optional[Sequence] = None,
        atom_desc_scaler=None,
    ):
        train_data, val_data, test_data = splitter.split(
            self.dataset, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac
        )

        # 1) snapshot raw once for each component of each sample
        for split in (train_data, val_data, test_data):
            for sample in split:
                for comp in sample:
                    _ensure_raw_single(comp)

        # 2) store transformers
        self.feature_transformers = feature_transformers
        self.target_transformers = target_transformers
        self.atom_desc_scaler = atom_desc_scaler
        self.n_targets_ = (
            len(target_transformers) if target_transformers is not None else None
        )

        # 3) FEATURES: fit per component on RAW train; transform from RAW
        if self.feature_transformers is not None:
            for idx, transformer in enumerate(self.feature_transformers):
                X = self._collect_features(train_data, idx)  # RAW
                transformer.fit(X)
            for split in (train_data, val_data, test_data):
                for idx, _ in enumerate(self.feature_transformers):
                    self._apply_feature_transform(split, idx)  # from RAW

        # 4) TARGETS: fit on RAW train from component 0; copy scaled y to all comps
        if self.target_transformers is not None:
            self._fit_target_transformers(train_data)  # RAW
            for split in (train_data, val_data, test_data):
                self._apply_target_transform(split)  # from RAW

        if self.atom_desc_scaler is not None:
            train_flat = [comp for sample in train_data for comp in sample]
            self.atom_desc_scaler.fit(train_flat)
            for split in (train_data, val_data, test_data):
                flat = [comp for sample in split for comp in sample]
                self.atom_desc_scaler.transform(flat)

        return train_data, val_data, test_data

    def _fit_target_transformers(self, train_data: List) -> None:
        Y = self._collect_targets(train_data)
        if self.n_targets_ is None:
            self.n_targets_ = Y.shape[1]
        assert self.n_targets_ == len(self.target_transformers)
        assert Y.shape[1] == self.n_targets_
        for i, t in enumerate(self.target_transformers):
            if (
                t.__class__.__name__ == "PowerTransformer"
                and getattr(t, "method", "") == "box-cox"
            ):
                if not np.all(Y[:, i] > 0):
                    bad = np.where(Y[:, i] <= 0)[0][:5]
                    raise ValueError(
                        f"Box–Cox column {i} has non-positive values at indices {bad.tolist()}"
                    )
            t.fit(Y[:, [i]])

    # Helpers ------------------------------------------------------------
    def _collect_features(self, data_list: List, comp_idx: int) -> np.ndarray:
        feats = [sample[comp_idx].global_features_raw.numpy() for sample in data_list]
        return np.vstack(feats)

    def _collect_extra_atom_features(
        self, data_list: List, comp_idx: int
    ) -> np.ndarray:
        extras = [
            d.f_atoms_raw[:, comp_idx].numpy()
            for d in data_list
            if hasattr(d, "f_atoms_raw")
        ]
        return np.vstack(extras) if len(extras) else np.zeros((0,))

    def _collect_targets(self, data_list: List) -> np.ndarray:
        targets = [sample[0].y_raw.numpy() for sample in data_list]
        return np.vstack(targets)

    def _apply_extra_atom_transformers(self, data_list: List) -> None:
        if self.extra_atom_transformers is None:
            return
        for d in data_list:
            for i, transformer in enumerate(self.extra_atom_transformers):
                if not hasattr(d, "f_atoms_raw") or d.f_atoms_raw is None:
                    continue
                x = d.f_atoms_raw[:, i].numpy().reshape(1, -1)
                transformed = transformer.transform(x)
                d.f_atoms[:, i] = torch.tensor(transformed, dtype=torch.float32)

    def _fit_extra_atom_transformers(self, train_data: List) -> None:
        if self.extra_atom_transformers is None:
            return
        for i, transformer in enumerate(self.extra_atom_transformers):
            X = self._collect_extra_atom_features(train_data, i)
            if X.size:
                transformer.fit(X)

    def _apply_feature_transform(self, data_list: List, comp_idx: int) -> None:
        transformer = self.feature_transformers[comp_idx]
        for sample in data_list:
            comp = sample[comp_idx]
            arr = comp.global_features_raw.numpy().reshape(1, -1)
            transformed = transformer.transform(arr)
            comp.global_features = torch.tensor(
                transformed.squeeze(), dtype=torch.float32
            )

    def _apply_target_transform(self, data_list: List) -> None:
        for sample in data_list:
            y = sample[0].y_raw.numpy().reshape(1, -1)
            assert y.shape[1] == self.n_targets_
            cols = [
                t.transform(y[:, [i]]) for i, t in enumerate(self.target_transformers)
            ]
            transformed = np.hstack(cols).reshape(-1)
            new_tensor = torch.tensor(transformed, dtype=torch.float32)
            for comp in sample:
                comp.y = new_tensor.clone()

    def _fit_extra_atom_transformers(self, train_data: List) -> None:
        if self.extra_atom_transformers is None:
            return
        for i, transformer in enumerate(self.extra_atom_transformers):
            X = self._collect_extra_atom_features(train_data, i)
            if X.size:
                transformer.fit(X)

    def _fit_target_transformers(self, train_data: List) -> None:
        Y = self._collect_targets(train_data)
        if self.n_targets_ is None:
            self.n_targets_ = Y.shape[1]
        assert self.n_targets_ == len(self.target_transformers)
        assert Y.shape[1] == self.n_targets_
        for i, t in enumerate(self.target_transformers):
            if (
                t.__class__.__name__ == "PowerTransformer"
                and getattr(t, "method", "") == "box-cox"
            ):
                if not np.all(Y[:, i] > 0):
                    bad = np.where(Y[:, i] <= 0)[0][:5]
                    raise ValueError(
                        f"Box–Cox column {i} has non-positive values at indices {bad.tolist()}"
                    )
            t.fit(Y[:, [i]])

    def build_target_unscaler(self):
        return (
            ColumnUnscaler(self.target_transformers)
            if self.target_transformers
            else None
        )

    def reset_to_raw(self, data_list: Optional[List] = None) -> None:
        if data_list is None:
            data_list = self.dataset
        for sample in data_list:
            for comp in sample:
                if hasattr(comp, "y_raw"):
                    comp.y = comp.y_raw.detach().clone()
                if hasattr(comp, "global_features_raw"):
                    comp.global_features = comp.global_features_raw.detach().clone()
                if hasattr(comp, "f_atoms_raw"):
                    comp.f_atoms = comp.f_atoms_raw.detach().clone()
                if hasattr(comp, "f_bonds_raw"):
                    comp.f_bonds = comp.f_bonds_raw.detach().clone()
