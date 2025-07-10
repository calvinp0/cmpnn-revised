import random
from collections import defaultdict
from typing import List, Tuple, Union

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import KFold

from cmpnn.split.base import BaseSplitter
from cmpnn.data.molecule_data import MoleculeData

class ScaffoldSplitter(BaseSplitter):
    def __init__(self, seed: int = 42):
        super().__init__(seed)

    def generate_scaffold(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return scaffold

    def _group_by_scaffold(self, dataset, component_idx: int = 0) -> List[List[int]]:
        scaffolds = defaultdict(list)
        for i, data in enumerate(dataset):
            try:
                smiles = self._get_smiles(data, i, component_idx)
                scaffold = self.generate_scaffold(smiles)
                scaffolds[scaffold].append(i)
            except Exception as e:
                print(f"Failed to process index {i}: {e}")
        scaffold_sets = list(scaffolds.values())
        scaffold_sets.sort(key=lambda x: len(x), reverse=True)
        return scaffold_sets

    def split(
            self,
            dataset,
            train_frac=0.8,
            val_frac=None,
            test_frac=None,
            return_indices=False,
            scaffold_component: str = 0,
    ):
        if val_frac is None and test_frac is None:
            raise ValueError("At least one of val_frac or test_frac must be specified.")
        if val_frac is None:
            val_frac = 0.0
        if test_frac is None:
            test_frac = 0.0

        total = train_frac + val_frac + test_frac
        assert abs(total - 1.0) < 1e-6, f"Fractions must sum to 1.0 (got {total})"

        if isinstance(scaffold_component, str):
            scaffold_component = {'donor': 0, 'acceptor': 1}.get(scaffold_component.lower(), 0)

        scaffold_sets = self._group_by_scaffold(dataset=dataset, component_idx=scaffold_component)

        train_idx, val_idx, test_idx = [], [], []
        n_total = len(dataset)
        n_train = int(train_frac * n_total)
        n_val = int(val_frac * n_total)
        n_test = n_total - n_train - n_val

        for group in scaffold_sets:
            if len(train_idx) + len(group) <= n_train:
                train_idx.extend(group)
            elif len(val_idx) + len(group) <= n_val:
                val_idx.extend(group)
            elif len(test_idx) + len(group) <= n_test:
                test_idx.extend(group)
            else:
                # If none can accept, assign to the smallest
                smallest = min([train_idx, val_idx, test_idx], key=len)
                smallest.extend(group)

        if val_frac > 0 and len(val_idx) < int(0.5 * val_frac * n_total):
            print(f"[WARNING] Validation set only contains {len(val_idx)} samples "
                f"but was expected to contain ~{int(val_frac * n_total)}. "
                "This suggests your scaffold distribution is highly imbalanced.")

        if test_frac > 0 and len(test_idx) < int(0.5 * test_frac * n_total):
            print(f"[WARNING] Test set only contains {len(test_idx)} samples "
                f"but was expected to contain ~{int(test_frac * n_total)}.")
    
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

    def _get_smiles(self, data: Union[MoleculeData, List[MoleculeData]], index: int, component: Union[int, str]) -> str:
        """
        Extracts the SMILES string from a MoleculeData or a list of MoleculeData.
        """
        # Case 1: List of MoleculeData (e.g., [donor, acceptor])
        if isinstance(data, list):
            if isinstance(component, int):
                return data[component].smiles
            elif isinstance(component, str):
                for m in data:
                    if hasattr(m, 'type') and m.type == component:
                        return m.smiles
                raise ValueError(f"No molecule in sample at index {index} has type='{component}'")
            else:
                raise TypeError(f"Unsupported component type: {type(component)}")

        # Case 2: single MoleculeData (e.g., MoleculeDataset)
        elif isinstance(data, MoleculeData):
            return data.smiles

        else:
            raise TypeError(f"Unsupported data format at index {index}: {type(data)}")