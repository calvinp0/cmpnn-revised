from typing import Callable

import torch
from torch.utils.data import DataLoader

from cmpnn.data.molecule_data import MoleculeDataBatch


class CrossValidator:
    def __init__(self,
                 dataset,
                 model_fn: Callable[[], 'pl.LightningModule'],
                 trainer_fn: Callable[[], 'pl.Trainer'],
                 splitter,
                 k: int = 5,
                 batch_size: int = 32,
                 num_workers: int = 0,
                 test_set=None):
        self.dataset = dataset
        self.model_fn = model_fn
        self.trainer_fn = trainer_fn
        self.splitter = splitter
        self.k = k
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_set = test_set

    def run(self):
        folds = self.splitter.k_fold_split(self.dataset, k=self.k)
        fold_metrics = []

        for fold_idx, (train_data, val_data) in enumerate(folds):
            print(f"Running Fold {fold_idx + 1}/{self.k}")

            train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True,
                                      num_workers=self.num_workers, collate_fn=MoleculeDataBatch.from_data_list)
            val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False,
                                    num_workers=self.num_workers, collate_fn=MoleculeDataBatch.from_data_list)

            model = self.model_fn()
            trainer = self.trainer_fn()

            trainer.fit(model, train_loader, val_loader)
            val_metrics = trainer.validate(model, val_loader)[0]
            fold_metrics.append(val_metrics)

        print("Cross-validation finished. Aggregating results...")
        for metric in fold_metrics[0].keys():
            values = [m[metric] for m in fold_metrics]
            print(f"{metric}: mean={sum(values) / len(values):.4f}, std={torch.std(torch.tensor(values)):.4f}")

        if self.test_set is not None:
            print("Retraining on full dataset and evaluating on held-out test set...")
            full_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                                     num_workers=self.num_workers, collate_fn=MoleculeDataBatch.from_data_list)
            test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False,
                                     num_workers=self.num_workers, collate_fn=MoleculeDataBatch.from_data_list)
            model = self.model_fn()
            trainer = self.trainer_fn()
            trainer.fit(model, full_loader)
            test_metrics = trainer.test(model, test_loader)[0]
            print("Held-out test metrics:", test_metrics)

        return fold_metrics
