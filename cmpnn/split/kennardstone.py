import numpy as np
from typing import Union, Tuple, List
import torch
from cmpnn.split.base import BaseSplitter
from sklearn.metrics import pairwise_distances
from cmpnn.data.molecule_data import MultiMoleculeDataBatch, MoleculeData
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns


class KennardStoneSplitter(BaseSplitter):
    def __init__(self, seed=42, component='joint', distance_mode='mean', distance_metric='euclidean'):
        super().__init__(seed)
        self.component = component
        self.distance_mode = distance_mode
        self.distance_metric = distance_metric

    def fast_kennard_stone(self, ks_distance: np.ndarray) -> np.ndarray:
        """
        Vectorized and correct Kennard-Stone selection.
        """
        n_samples = ks_distance.shape[0]
        np.fill_diagonal(ks_distance, -np.inf)

        # Step 1: find the two most distant samples
        i1, i2 = np.unravel_index(np.nanargmax(ks_distance), ks_distance.shape)
        selected = [i1, i2]

        order = np.full(n_samples, -1, dtype=int)
        order[0], order[1] = i1, i2

        mask = np.ones(n_samples, dtype=bool)
        mask[i1] = False
        mask[i2] = False

        # Track min distance of each point to selected
        min_dist = np.minimum(ks_distance[:, i1], ks_distance[:, i2])

        for step in range(2, n_samples):
            min_dist[~mask] = -np.inf
            next_idx = np.argmax(min_dist)

            order[step] = next_idx
            mask[next_idx] = False

            # Update distances
            min_dist = np.minimum(min_dist, ks_distance[:, next_idx])

        return order

    
    def split(
        self,
        dataset: MultiMoleculeDataBatch,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        return_indices: bool = False,
        return_order: bool = False,
    ) -> Union[
        Tuple[List, List, List],
        Tuple[List[int], List[int], List[int]]
    ]:
        dist = self.compute_distance_matrix(dataset)
        print("Distance matrix shape:", dist.shape)
        print("Distance matrix min/max:", dist.min(), dist.max())
        print("Distance matrix diagonal:", np.diag(dist)[:10])
        order = self.fast_kennard_stone(dist)

        n = len(dataset)
        n_train = int(train_frac * n)
        n_val = int(val_frac * n)
        n_test = n - n_train - n_val

        train_idx = order[:n_train].tolist()
        val_idx = order[n_train:n_train + n_val].tolist()
        test_idx = order[n_train + n_val:].tolist()

        if return_order and return_indices:
                return (
                    [dataset[i] for i in train_idx],
                    [dataset[i] for i in val_idx],
                    [dataset[i] for i in test_idx],
                    train_idx,
                    val_idx,
                    test_idx,
                    order,
                )
        elif return_order:
            return [dataset[i] for i in train_idx], [dataset[i] for i in val_idx], [dataset[i] for i in test_idx], order
        elif return_indices:
            return train_idx, val_idx, test_idx
        else:
            return [dataset[i] for i in train_idx], [dataset[i] for i in val_idx], [dataset[i] for i in test_idx]



    def get_padded_embeddings(self, mols: List[MoleculeData]) -> np.ndarray:
        """
        Returns padded atom features for a list of MoleculeData objects.
        Shape: [n_samples, max_atoms, f_dim] → flattened to [n_samples, max_atoms * f_dim]
        """
        f_list = [mol.f_atoms.detach().cpu() for mol in mols]
        max_atoms = max(f.shape[0] for f in f_list)
        f_dim = f_list[0].shape[1]

        padded = torch.zeros((len(f_list), max_atoms, f_dim), dtype=torch.float32)
        for i, f in enumerate(f_list):
            padded[i, :f.shape[0], :] = f

        # Optional: also return mask
        # mask = torch.zeros((len(f_list), max_atoms), dtype=torch.bool)
        # for i, f in enumerate(f_list):
        #     mask[i, :f.shape[0]] = 1

        return padded.view(len(f_list), -1)  # Flatten to [n_samples, max_atoms * f_dim]
        
    def compute_distance_matrix(self, dataset: MultiMoleculeDataBatch) -> np.ndarray:
        """
        Auto-computes the distance matrix based on splitter config.
        """
        donors = [pair[0] for pair in dataset]
        acceptors = [pair[1] for pair in dataset]

        donor_embs = self.get_padded_embeddings(donors)
        acceptor_embs = self.get_padded_embeddings(acceptors)

        if self.component in ['donor', 0]:
            return pairwise_distances(donor_embs, metric=self.distance_metric)

        elif self.component in ['acceptor', 1]:
            return pairwise_distances(acceptor_embs, metric=self.distance_metric)

        elif self.component == 'joint':
            if self.distance_mode == 'mean':
                dist1 = pairwise_distances(donor_embs, metric=self.distance_metric)
                dist2 = pairwise_distances(acceptor_embs, metric=self.distance_metric)
                
                return 0.5 * dist1 + 0.5 * dist2

            elif self.distance_mode == 'max':
                dist1 = pairwise_distances(donor_embs, metric=self.distance_metric)
                dist2 = pairwise_distances(acceptor_embs, metric=self.distance_metric)
                return np.maximum(dist1, dist2)

            elif self.distance_mode == 'concat':
                joint_emb = np.concatenate([donor_embs, acceptor_embs], axis=1)
                return pairwise_distances(joint_emb, metric=self.distance_metric)

            else:
                raise ValueError(f"Unknown joint distance_mode: {self.distance_mode}")

        else:
            raise ValueError(f"Invalid component: {self.component}")

    def plot_distance_order(self, dataset, order: List[int], title="Kennard-Stone Order", method='pca'):
        """
        Visualize selection order via PCA or t-SNE projection.
        Assumes `dataset` is a list of [donor, acceptor] MoleculeData pairs.
        """
        # Only use the raw dataset — don't pass in a batch or loader
        data_ordered = [dataset[i] for i in order]

        donor_embs = self.get_padded_embeddings([pair[0] for pair in data_ordered])
        acceptor_embs = self.get_padded_embeddings([pair[1] for pair in data_ordered])
        joint_embs = np.concatenate([donor_embs, acceptor_embs], axis=1)

        # Dimensionality reduction
        if method == 'pca':
            reducer = PCA(n_components=2)
        else:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2)

        coords = reducer.fit_transform(joint_embs)

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=np.arange(len(order)), cmap='viridis', s=50)
        plt.colorbar(scatter, label='KS Selection Step')
        plt.title(title)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
