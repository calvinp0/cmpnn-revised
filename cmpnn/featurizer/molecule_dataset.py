import os
from typing import List, Optional, Union, Dict
import pandas as pd
import numpy as np
from rdkit import Chem
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Dataset
from cmpnn.data.molecule_data import MoleculeData, MultiMoleculeDataBatch
from cmpnn.featurizer.utils import featurize_molecule, infer_dtype
from sklearn.preprocessing import LabelEncoder
import pickle


class MoleculeDataset(InMemoryDataset):
    """
    MoleculeDataset is a subclass of InMemoryDataset that is used to handle molecular data.
    It provides methods to load, process, and transform molecular data into a format suitable for machine learning tasks.
    """

    def __init__(self, csv_file: str, atom_featurizer, bond_featurizer, atom_messages: bool = False,
                 transform=None, pre_transform=None, global_featurizer=None, use_cache: bool = True,
                 weights_only: bool = False, smiles_col: str = "smiles", target_cols: Union[List[str], Dict[str, torch.dtype]] = None, label_encoder_path: Optional[str] = None,
                 atom_descriptor_csv: str = None, rxn_col: str = "rxn_id", mol_type_col: str = "mol_type",
                 **kwargs):

        self.csv_file = csv_file
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer
        self.atom_messages = atom_messages
        self.global_featurizer = global_featurizer
        self.use_cache = use_cache
        self.kwargs = kwargs
        self.smiles_col = smiles_col
        self.target_cols = target_cols
        self.label_encoders = {}
        self.atom_descriptor_df = pd.read_csv(atom_descriptor_csv) if atom_descriptor_csv else None
        self.rxn_col = rxn_col
        self.mol_type_col = mol_type_col
        if self.atom_descriptor_df is not None:
            self.descriptor_cols = [c for c in self.atom_descriptor_df.columns if c not in [self.rxn_col, self.mol_type_col]]
        else:
            self.descriptor_cols = []
        super().__init__('.', transform, pre_transform)

        if self.use_cache and os.path.exists(self.processed_paths[0]):
            print(f"Loading cached data from {self.processed_paths[0]}")
            self._data, self.slices = torch.load(self.processed_paths[0], weights_only=weights_only)
            
            # Load encoders if path provided
            if label_encoder_path and os.path.exists(label_encoder_path):
                self.load_label_encoders(label_encoder_path)
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
        pass

    def process(self) -> List[MoleculeData]:
        df = pd.read_csv(self.csv_file)
        
        if isinstance(self.target_cols, dict):
            target_map = self.target_cols
        elif isinstance(self.target_cols, list):
            target_map = {col: infer_dtype(df[col]) for col in self.target_cols}
        else:
            raise ValueError("target_cols must be a list or a dictionary")

        # fit encoders for string labels TODO: Need to work on this at a later date
        for col, dtype in target_map.items():
            if dtype == torch.long and df[col].dtype == object:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le

        data_list = []
        
        for idx, row in df.iterrows():
            smiles = row[self.smiles_col]
            target = [torch.tensor(row[col], dtype=target_map[col]) for col in target_map]
            y = torch.stack(target) if len(target) > 1 else target[0]

            extra_features = None
            if self.atom_descriptor_df is not None and self.rxn_col in row and self.mol_type_col in row:
                rxn_id = row[self.rxn_col]
                mol_type = row[self.mol_type_col]
                mask = (self.atom_descriptor_df[self.rxn_col] == rxn_id) & (self.atom_descriptor_df[self.mol_type_col] == mol_type)
                sub_df = self.atom_descriptor_df[mask]
                if not sub_df.empty:
                    extra_features = sub_df[self.descriptor_cols].values

            data = featurize_molecule(smiles=smiles, target=y,
                                      atom_featurizer=self.atom_featurizer,
                                      bond_featurizer=self.bond_featurizer,
                                      global_featurizer=self.global_featurizer,
                                      atom_messages=self.atom_messages,
                                      extra_atom_features=extra_features)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        return data_list

    def save_label_encoders(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.label_encoders, f)

    def load_label_encoders(self, path):
        with open(path, 'rb') as f:
            self.label_encoders = pickle.load(f)




class MultiMoleculeDataset(Dataset):
    """
    MultiMoleculeDataset is a subclass of Dataset that is used to handle multiple molecular datasets.
    It provides methods to load, process, and transform molecular data into a format suitable for machine learning tasks.
    """
    def __init__(self, csv_file: str, atom_featurizer, bond_featurizer, global_featurizer=None, cache_path: str = None,
                 atom_messages: bool = False, use_cache: bool = True, weights_only: bool = False,
                 smiles_cols: List[str] = ["smiles1", "smiles2"], target_cols: Union[List[str], Dict[str, torch.dtype]] = None, label_encoder_path: Optional[str] = None, atom_descriptor_csv: str = None, rxn_col: str = "rxn_id", mol_type_cols: Optional[List[str]] = None, mol_type_col: str = "mol_type"):

        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer
        self.global_featurizer = global_featurizer
        self.cache_path = cache_path or csv_file.replace(".csv", "_cache.pt")
        self.atom_messages = atom_messages
        self.use_cache = use_cache
        self.smiles_cols = smiles_cols
        self.target_cols = target_cols
        self.label_encoders = {}
        self.atom_descriptor_df = pd.read_csv(atom_descriptor_csv) if atom_descriptor_csv else None
        self.rxn_col = rxn_col
        self.mol_type_cols = mol_type_cols or []
        self.desc_mol_type_col = mol_type_col
        if self.atom_descriptor_df is not None:
            self.descriptor_cols = [c for c in self.atom_descriptor_df.columns if c not in [self.rxn_col, self.desc_mol_type_col]]
        else:
            self.descriptor_cols = []
        if os.path.exists(self.cache_path) and self.use_cache:
            print(f"Loading cached dataset from {self.cache_path}")
            self.data = torch.load(self.cache_path, weights_only=weights_only)

            if label_encoder_path and os.path.exists(label_encoder_path):
                self.load_label_encoders(label_encoder_path)
        else:
            print(f"Processing and caching dataset to {self.cache_path}")
            self.data = self._process(csv_file)
            torch.save(self.data, self.cache_path)

    def _process(self, csv_file: str) -> List[List[MoleculeData]]:
        df = pd.read_csv(csv_file)
        data_list = []

        if isinstance(self.target_cols, dict):
            target_map = self.target_cols
        elif isinstance(self.target_cols, list):
            target_map = {col: infer_dtype(df[col]) for col in self.target_cols}
        else:
            raise ValueError("target_cols must be a list or a dictionary")

        for col, dtype in target_map.items():
            if dtype == torch.long and df[col].dtype == object:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le

        for _, row in df.iterrows():
            smiles = [row[col] for col in self.smiles_cols]
            mol_types = [row[col] for col in self.mol_type_cols] if self.mol_type_cols else [None] * len(smiles)
            target = [torch.tensor(row[col], dtype=target_map[col]) for col in target_map]
            y = torch.stack(target) if len(target) > 1 else target[0]

            mols_rdkit = [Chem.MolFromSmiles(smi) for smi in smiles]
            if any(mol is None for mol in mols_rdkit):
                continue

            rxn_id = row[self.rxn_col] if self.rxn_col in row else None
            data = []
            for smi, mol_type in zip(smiles, mol_types):
                extra_features = None
                if self.atom_descriptor_df is not None and rxn_id is not None:
                    mask = self.atom_descriptor_df[self.rxn_col] == rxn_id
                    if mol_type is not None:
                        mask &= self.atom_descriptor_df[self.desc_mol_type_col] == mol_type
                    sub_df = self.atom_descriptor_df[mask]
                    if not sub_df.empty:
                        extra_features = sub_df[self.descriptor_cols].values
                data.append(featurize_molecule(smi, y, self.atom_featurizer, self.bond_featurizer,
                                              self.global_featurizer, self.atom_messages,
                                              extra_atom_features=extra_features))
            data_list.append(data)

        return data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def save_label_encoders(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.label_encoders, f)

    def load_label_encoders(self, path):
        with open(path, 'rb') as f:
            self.label_encoders = pickle.load(f)
