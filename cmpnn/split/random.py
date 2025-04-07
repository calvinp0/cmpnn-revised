import random
from cmpnn.split.base import BaseSplitter


class RandomSplitter(BaseSplitter):
    def split(self, dataset, train_frac=0.8, val_frac=0.1, test_frac=0.1):
        assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

        rng = random.Random(self.seed)
        indices = list(range(len(dataset)))
        rng.shuffle(indices)

        n_total = len(dataset)
        n_train = int(train_frac * n_total)
        n_val = int(val_frac * n_total)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        train_data = [dataset[i] for i in train_idx]
        val_data = [dataset[i] for i in val_idx]
        test_data = [dataset[i] for i in test_idx]

        return train_data, val_data, test_data
