from collections import defaultdict
from typing import List, Tuple, Union
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import random
from cmpnn.split.base import BaseSplitter
from sklearn.model_selection import KFold


class ScaffoldSplitter(BaseSplitter):
    def __init__(self, seed: int = 42):
        super().__init__(seed)

    def generate_scaffold(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return scaffold

    def _group_by_scaffold(self, dataset) -> List[List[int]]:
        scaffolds = defaultdict(list)
        for i, data in enumerate(dataset):
            scaffold = self.generate_scaffold(data.smiles)
            scaffolds[scaffold].append(i)
        scaffold_sets = list(scaffolds.values())
        scaffold_sets.sort(key=lambda x: len(x), reverse=True)
        return scaffold_sets

    def split(
        self,
        dataset,
        train_frac=0.8,
        val_frac=None,
        test_frac=None,
        return_indices=False
    ):
        if val_frac is None and test_frac is None:
            raise ValueError("At least one of val_frac or test_frac must be specified.")
        if val_frac is None:
            val_frac = 0.0
        if test_frac is None:
            test_frac = 0.0

        total = train_frac + val_frac + test_frac
        assert abs(total - 1.0) < 1e-6, f"Fractions must sum to 1.0 (got {total})"

        scaffold_sets = self._group_by_scaffold(dataset)

        train_idx, val_idx, test_idx = [], [], []
        n_total = len(dataset)
        n_train = int(train_frac * n_total)
        n_val = int(val_frac * n_total)

        rng = random.Random(self.seed)

        for group in scaffold_sets:
            if len(train_idx) + len(group) <= n_train:
                train_idx.extend(group)
            elif len(val_idx) + len(group) <= n_val:
                val_idx.extend(group)
            else:
                test_idx.extend(group)

        if return_indices:
            if val_frac > 0:
                return train_idx, val_idx, test_idx
            else:
                return train_idx, test_idx
        else:
            if val_frac > 0:
                return [dataset[i] for i in train_idx], [dataset[i] for i in val_idx], [dataset[i] for i in test_idx]
            else:
                return [dataset[i] for i in train_idx], [dataset[i] for i in test_idx]

    def k_fold_split(self, dataset, k: int = 5, shuffle=True, return_indices=False) -> List[Union[Tuple, List]]:
        scaffold_sets = self._group_by_scaffold(dataset)

        # Flatten and preserve scaffold groupings for folding
        all_indices = []
        for group in scaffold_sets:
            all_indices.extend(group)

        if shuffle:
            random.Random(self.seed).shuffle(all_indices)

        kf = KFold(n_splits=k, shuffle=False)
        folds = []
        for train_index, test_index in kf.split(all_indices):
            train_idx = [all_indices[i] for i in train_index]
            test_idx = [all_indices[i] for i in test_index]
            if return_indices:
                folds.append((train_idx, test_idx))
            else:
                folds.append((
                    [dataset[i] for i in train_idx],
                    [dataset[i] for i in test_idx]
                ))
        return folds
