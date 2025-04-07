import os
from typing import List, Optional
import pandas as pd
import numpy as np
from rdkit import Chem
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Dataset
from cmpnn.data.molecule_data import MoleculeData, MultiMoleculeDataBatch
from cmpnn.featurizer.utils import featurize_molecule

class MoleculeDataset(InMemoryDataset):
    """
    MoleculeDataset is a subclass of InMemoryDataset that is used to handle molecular data.
    It provides methods to load, process, and transform molecular data into a format suitable for machine learning tasks.
    """

    def __init__(self, csv_file: str, atom_featurizer: callable, bond_featurizer: callable, atom_messages: bool = False,
                 transform=None, pre_transform=None, global_featurizer: callable = None, use_cache: bool = True, weights_only: bool = False, **kwargs):
        
        self.csv_file = csv_file
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer
        self.atom_messages = atom_messages
        self.global_featurizer = global_featurizer
        self.use_cache = use_cache
        self.kwargs = kwargs
        super().__init__('.', transform, pre_transform)
        if self.use_cache and os.path.exists(self.processed_paths[0]):
            print(f"Loading cached data from {self.processed_paths[0]}")
            self._data, self.slices = torch.load(self.processed_paths[0], weights_only=weights_only)
        else:
            print(f"Processing data from {self.csv_file}")
            data_list = self.process()
            self._data, self.slices = self.collate(data_list)
            torch.save((self._data, self.slices), self.processed_paths[0])
    @property
    def raw_file_names(self):
        return [os.path.basename(self.csv_file)]
    
    @property
    def processed_file_names(self):
        return ['molecule_data.pt']
    
    def download(self):
        # Implement the download logic if needed
        pass

    def process(self) -> List[MoleculeData]:
        """
        Process the raw data and convert it into a list of MoleculeData objects.
        """
        # Load the CSV file
        df = pd.read_csv(self.csv_file)
        data_list = []

        for idx, row in df.iterrows():
            smiles = row['smiles']
            try:
                target = row['target']
            except ValueError as e:
                print(f"Error processing target for SMILES {smiles}: {e}")
                target = None
    
            data = featurize_molecule(smiles=smiles, target=target,
                                      atom_featurizer=self.atom_featurizer,
                                      bond_featurizer=self.bond_featurizer,
                                      global_featurizer=self.global_featurizer,
                                      atom_messages=self.atom_messages)

            data_list.append(data)

        # Save the processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        return data_list


class MultiMoleculeDataset(Dataset):
    def __init__(self, csv_file: str, atom_featurizer, bond_featurizer, global_featurizer=None, cache_path: str = None,
                 atom_messages: bool = False, use_cache: bool = True):
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer
        self.global_featurizer = global_featurizer
        self.cache_path = cache_path or csv_file.replace(".csv", "_cache.pt")
        self.atom_messages = atom_messages
        self.use_cache = use_cache

        if os.path.exists(self.cache_path):
            print(f"Loading cached dataset from {self.cache_path}")
            self.data = torch.load(self.cache_path, weights_only=False)
        else:
            print(f"Processing and caching dataset to {self.cache_path}")
            self.data = self._process(csv_file)
            torch.save(self.data, self.cache_path)

    def _process(self, csv_file: str) -> List[List[MoleculeData]]:
        df = pd.read_csv(csv_file)
        data_list = []
        for _, row in df.iterrows():
            smiles1, smiles2 = row['smiles1'], row['smiles2']
            target = torch.tensor([row['target']], dtype=torch.float)

            mol1, mol2 = Chem.MolFromSmiles(smiles1), Chem.MolFromSmiles(smiles2)
            if mol1 is None or mol2 is None:
                continue

            data1 = featurize_molecule(smiles1, target, self.atom_featurizer, self.bond_featurizer, self.global_featurizer, self.atom_messages)
            data2 = featurize_molecule(smiles2, target, self.atom_featurizer, self.bond_featurizer, self.global_featurizer, self.atom_messages)

            data_list.append([data1, data2])
        return data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
