import numpy as np
import logging
from typing import List, Set, Tuple
import matplotlib.pyplot as plt


def log_scaffold_stats(data, 
                       index_sets: List[Set[int]],
                       num_scaffolds: int = 10,
                       num_labels: int = 20,
                       logger: logging.Logger = None
                      ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Logs and returns statistics about counts, means, and standard deviations in molecular scaffolds.

    :param data: A MoleculeDataset.
    :param index_sets: A list of sets of indices (e.g., train, val, test).
    :param num_scaffolds: Max number of splits to report.
    :param num_labels: Max number of labels per molecule.
    :param logger: Optional logger.
    :return: A list of (mean, std, count) per split.
    """
    target_means, target_stds, counts = [], [], []

    for index_set in index_sets:
        targets = [data[idx].y.cpu().numpy() for idx in index_set if hasattr(data[idx], 'y')]
        if len(targets) == 0:
            target_means.append(np.array([np.nan] * num_labels))
            target_stds.append(np.array([np.nan] * num_labels))
            counts.append(np.array([0] * num_labels))
            continue

        targets = np.array(targets, dtype=np.float32)
        if targets.ndim == 1:
            targets = targets[:, np.newaxis]

        mean_targets = np.nanmean(targets, axis=0)
        std_targets = np.nanstd(targets, axis=0)
        count_targets = np.count_nonzero(~np.isnan(targets), axis=0)

        target_means.append(mean_targets[:num_labels])
        target_stds.append(std_targets[:num_labels])
        counts.append(count_targets[:num_labels])

    if logger is not None:
        logger.info('Label mean/std/count per split (max %d splits, %d labels):', num_scaffolds, num_labels)
        for i, (mean, std, count) in enumerate(zip(target_means, target_stds, counts)):
            logger.info(f"Split {i}: mean={mean}, std={std}, count={count}")

    return list(zip(target_means, target_stds, counts))


def plot_split_distributions(dataset, index_sets, label_index: int = 0):
    """
    Plot distribution of a specific label (by index) across splits.
    """
    labels = ['Train', 'Validation', 'Test']
    values = []

    for split_idx, idx_set in enumerate(index_sets):
        y_vals = [dataset[i].y[label_index].item() for i in idx_set if dataset[i].y is not None]
        values.append(y_vals)

    # Histogram
    plt.figure(figsize=(8, 5))
    for i, vals in enumerate(values):
        plt.hist(vals, bins=20, alpha=0.6, label=labels[i])
    plt.title(f'Histogram of Label {label_index} Across Splits')
    plt.xlabel('Target Value')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Boxplot
    plt.figure(figsize=(6, 4))
    plt.boxplot(values, labels=labels)
    plt.title(f'Boxplot of Label {label_index} Across Splits')
    plt.ylabel('Target Value')
    plt.tight_layout()
    plt.show()