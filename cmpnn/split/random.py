import random
from typing import List, Tuple, Union

from cmpnn.split.base import BaseSplitter

class RandomSplitter(BaseSplitter):
    def __init__(self, seed: int = 42):
        super().__init__(seed)

    def split(
        self,
        dataset,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        return_indices: bool = False,
        return_order: bool = False,
    ) -> Union[
        Tuple[List, List, List],
        Tuple[List[int], List[int], List[int]],
        Tuple[List, List, List, List[int]],
        Tuple[List, List, List, List[int], List[int], List[int]],
    ]:
        assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, \
            f"Fractions must sum to 1, got {train_frac + val_frac + test_frac}"

        rng = random.Random(self.seed)
        indices = list(range(len(dataset)))
        rng.shuffle(indices)

        n_total = len(dataset)
        n_train = int(train_frac * n_total)
        n_val = int(val_frac * n_total)
        n_test = n_total - n_train - n_val

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        if return_order and return_indices:
            return (
                [dataset[i] for i in train_idx],
                [dataset[i] for i in val_idx],
                [dataset[i] for i in test_idx],
                train_idx,
                val_idx,
                test_idx,
                indices,
            )
        elif return_order:
            return (
                [dataset[i] for i in train_idx],
                [dataset[i] for i in val_idx],
                [dataset[i] for i in test_idx],
                indices,
            )
        elif return_indices:
            return train_idx, val_idx, test_idx
        else:
            return (
                [dataset[i] for i in train_idx],
                [dataset[i] for i in val_idx],
                [dataset[i] for i in test_idx]
            )
    