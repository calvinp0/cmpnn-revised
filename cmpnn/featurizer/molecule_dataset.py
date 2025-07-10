import os
import pickle
from typing import List, Optional, Union, Dict
import json

import pandas as pd
import torch
from rdkit import Chem
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Dataset
from torch_geometric.data import InMemoryDataset

from cmpnn.data.molecule_data import MoleculeData
from cmpnn.data.target import TargetBuilder, TargetSpec
from cmpnn.featurizer.utils import featurize_molecule, infer_dtype
from cmpnn.standardisation.target_normalizer import TargetNormalizer


class MoleculeDataset(InMemoryDataset):
    """
    MoleculeDataset is a subclass of InMemoryDataset that is used to handle molecular data.
    It provides methods to load, process, and transform molecular data into a format suitable for machine learning tasks.
    """

    def __init__(self, csv_file: str, atom_featurizer, bond_featurizer, atom_messages: bool = False,
                 transform=None, pre_transform=None, global_featurizer=None, use_cache: bool = True,
                 weights_only: bool = False, smiles_col: str = "smiles", add_hs: bool = False, sanitize: bool = True,
                 target_cols: Union[List[str], Dict[str, torch.dtype]] = None, label_encoder_path: Optional[str] = None, normalizer: callable = None,
                 **kwargs):

        self.csv_file = csv_file
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer
        self.atom_messages = atom_messages
        self.add_hs = add_hs
        self.sanitize = sanitize
        self.global_featurizer = global_featurizer
        self.use_cache = use_cache
        self.kwargs = kwargs
        self.smiles_col = smiles_col
        self.target_cols = target_cols
        self.label_encoders = {}
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

        if normalizer:
            data_list = [self.get(i) for i in range(len(self))]
            ys = torch.stack([d.y for d in data_list])
            self.normalizer = normalizer
            self.normalizer.fit(ys)
            for d in data_list:
                d.y = self.normalizer.transform(d.y)
            self._data, self.slices = self.collate(data_list)

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
            data = featurize_molecule(mol_or_smiles=smiles, target=y,
                                      atom_featurizer=self.atom_featurizer,
                                      bond_featurizer=self.bond_featurizer,
                                      global_featurizer=self.global_featurizer,
                                      atom_messages=self.atom_messages,
                                      add_hs=self.add_hs, sanitize=self.sanitize)
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

    def apply_transform_to_y(self, tranform_fn):
        """
        Apply a transformation function to the target variable y.
        :param tranform_fn: A function that takes a tensor and returns a transformed tensor.
        """
        data_list = [self.get(i) for i in range(len(self))]
        for d in data_list:
            d.y = tranform_fn(d.y)
        self._data, self.slices = self.collate(data_list)


class MultiMoleculeDataset(Dataset):
    """
    MultiMoleculeDataset is a subclass of Dataset that is used to handle multiple molecular datasets.
    It provides methods to load, process, and transform molecular data into a format suitable for machine learning tasks.
    """

    def __init__(self, csv_file: str, atom_featurizer, bond_featurizer, add_hs: bool = False, sanitize: bool = True, global_featurizer=None, cache_path: str = None,
                 atom_messages: bool = False, use_cache: bool = True, weights_only: bool = False,
                 smiles_cols: List[str] = ["smiles1", "smiles2"], 
                 target_cols: Union[List[str], Dict[str, torch.dtype]] = None, normalizer: callable = None,
                 label_encoder_path: Optional[str] = None):

        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer
        self.add_hs = add_hs
        self.sanitize = sanitize
        self.global_featurizer = global_featurizer
        self.cache_path = cache_path or csv_file.replace(".csv", "_cache.pt")
        self.atom_messages = atom_messages
        self.use_cache = use_cache
        self.smiles_cols = smiles_cols
        self.target_cols = target_cols
        self.label_encoders = {}
        if os.path.exists(self.cache_path) and self.use_cache:
            print(f"Loading cached dataset from {self.cache_path}")
            self.data = torch.load(self.cache_path, weights_only=weights_only)

            if label_encoder_path and os.path.exists(label_encoder_path):
                self.load_label_encoders(label_encoder_path)
        else:
            print(f"Processing and caching dataset to {self.cache_path}")
            self.data = self._process(csv_file)
            torch.save(self.data, self.cache_path)

        if normalizer:
            ys = torch.stack([d[0].y for d in self.data])  # Assume both donor/acceptor have same target
            self.normalizer = normalizer
            self.normalizer.fit(ys)
            for pair in self.data:
                for d in pair:
                    d.y = self.normalizer.transform(d.y)

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
            target = [torch.tensor(row[col], dtype=target_map[col]) for col in target_map]
            y = torch.stack(target) if len(target) > 1 else target[0]

            mols = [Chem.MolFromSmiles(smi) for smi in smiles]
            if any(mol is None for mol in mols):
                continue

            data = [featurize_molecule(smi, y, self.atom_featurizer, self.bond_featurizer,
                                       self.global_featurizer, self.atom_messages, add_hs=self.add_hs, sanitize=self.sanitize) for smi in smiles]
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


class MoleculeSDFDataset(InMemoryDataset):
    
    def __init__(self, sdf_files: str, input_type: str, target_data: str, target_columns: List[str],
                 atom_featurizer, bond_featurizer, atom_messages: bool = False, normalizer: callable = None,
                 transform=None, pre_transform=None, global_featurizer=None, use_cache: bool = True,
                 weights_only: bool = False, add_hs: bool = False, sanitize: bool = True, keep_hs: bool = True, force_reload: bool = False, use_electro_map: bool = False):
        
        self.sdf_files = sdf_files
        self.input_type = input_type
        self.target_data = pd.read_csv(target_data)
        self.target_columns = target_columns
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer
        self.atom_messages = atom_messages
        self.add_hs = add_hs
        self.sanitize = sanitize
        self.keep_hs = keep_hs
        self.global_featurizer = global_featurizer
        self.use_cache = use_cache
        self.weights_only = weights_only
        self.use_electro_map = use_electro_map
        super().__init__('.', transform, pre_transform)
        # After the base class __init__ returns, _data and slices should be loaded or processed.
        if self.use_cache and os.path.exists(self.processed_paths[0]):
            print(f"Loading cached data from {self.processed_paths[0]}")
            self._data, self.slices = torch.load(self.processed_paths[0], weights_only=weights_only)

        else:
            print(f"Processing data from {self.sdf_files}")
            data_list = self.process()
            self._data, self.slices = self.collate(data_list)
            torch.save((self._data, self.slices), self.processed_paths[0])

        if normalizer:
            data_list = [self.get(i) for i in range(len(self))]
            ys = torch.stack([d.y for d in data_list])
            self.normalizer = normalizer
            self.normalizer.fit(ys)
            for d in data_list:
                d.y = self.normalizer.transform(d.y)
            self._data, self.slices = self.collate(data_list)

    @property
    def raw_file_names(self):
        if self.sdf_files.endswith('.sdf'):
            return [os.path.basename(self.sdf_files)]
        else:
            return [f for f in os.listdir(self.sdf_files) if f.endswith('.sdf')]


    @property
    def processed_file_names(self):
        return ['molecule_data_sdf.pt']
    
    def download(self):
        pass

    def process(self) -> List[MoleculeData]:

        if self.sdf_files.endswith('.sdf'):
            sdf_files = [self.sdf_files]
        else:
            sdf_files = [os.path.join(self.sdf_files, f) for f in os.listdir(self.sdf_files) if f.endswith('.sdf')]

        data_list = []

        for sdf_file in sdf_files:
            supplier = Chem.SDMolSupplier(sdf_file, removeHs=not self.keep_hs, sanitize=self.sanitize)

            mol = [m for m in supplier if m.GetProp('type') == self.input_type]
            if not mol:
                print(f"No valid molecules found in {sdf_file}")
                continue
            if len(mol) > 1:
                print(f"Multiple molecules found in {sdf_file}, pleasing check your sdf file.")
                continue
            mol = mol[0]
            if mol is None:
                print(f"Invalid molecule in {sdf_file}")
                continue

            if self.add_hs:
                mol = Chem.AddHs(mol)

            name = mol.GetProp('reaction')
            mol_properties = mol.GetProp("mol_properties") if mol.HasProp("mol_properties") else "{}"
            mol_props = json.loads(mol_properties)
            if not isinstance(mol_props, dict):
                print(f"Invalid mol_properties in {sdf_file}")
                continue
            smiles = Chem.MolToSmiles(mol)
            target_row = self.target_data[self.target_data['rxn'] == name]
            if target_row.empty:
                print(f"No target data found for {name}")
                continue
            target = [torch.tensor(target_row[col].values[0], dtype=torch.float) for col in self.target_columns]
            y = torch.stack(target) if len(target) > 1 else target[0]

            data = featurize_molecule(mol_or_smiles = mol, target=y,
                                      atom_featurizer=self.atom_featurizer,
                                      bond_featurizer=self.bond_featurizer,
                                      global_featurizer=self.global_featurizer,
                                      atom_messages=self.atom_messages,
                                    add_hs=self.add_hs, sanitize=self.sanitize, mol_properties=mol_props)

            data_list.append(data)
        self.data_count = len(data_list)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        return data_list
    
    def __len__(self):
        return self.data_count



class MultiMoleculeSDFDataset(Dataset):
    """
    MultiMoleculeSDFDataset is a subclass of Dataset that is used to handle multiple molecular datasets in SDF format.
    It provides methods to load, process, and transform molecular data into a format suitable for machine learning tasks.
    """

    def __init__(self, sdf_files: str, input_type: List[str], target_data: str, target_columns: List[str],
                 atom_featurizer, bond_featurizer, atom_messages: bool = False, normalizer: callable = None, atom_feature_normalizer: callable = None,
                 transform=None, pre_transform=None, global_featurizer=None, use_cache: bool = True, cache_path: str = None,
                 weights_only: bool = False, add_hs: bool = False, sanitize: bool = True, keep_hs: bool = True, force_reload: bool = False, use_electro_map: bool = False):
        
        self.sdf_files = sdf_files
        self.input_type = input_type
        self.target_data = pd.read_csv(target_data)
        self.target_columns = target_columns
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer
        self.atom_messages = atom_messages
        self.add_hs = add_hs
        self.sanitize = sanitize
        self.global_featurizer = global_featurizer
        self.use_cache = use_cache
        self.weights_only = weights_only
        self.keep_hs = keep_hs
        self.normalizer_fitted = False
        self.use_electro_map = use_electro_map
        self.atom_feature_normalizer = atom_feature_normalizer
        self.cache_path = cache_path or (
            sdf_files.replace(".sdf", "_cache.pt") if sdf_files.endswith(".sdf") 
            else os.path.join(sdf_files, "sdf_folder_cache.pt")
        )
        super().__init__('.', transform, pre_transform, force_reload=force_reload)

        if os.path.exists(self.cache_path) and self.use_cache:
            print(f"Loading cached dataset from {self.cache_path}")
            self.data = torch.load(self.cache_path, weights_only=weights_only)
        else:
            print(f"Processing and caching dataset to {self.cache_path}")
            self.data = self._process(sdf_files)
            torch.save(self.data, self.cache_path)

        self.normalizer = normalizer  # Just store, don't use yet
        self.normalizer_fitted = False

    def _process(self, sdf_files: str) -> List[List[MoleculeData]]:
        sdf_file_list = (
            [sdf_files] if sdf_files.endswith('.sdf')
            else sorted(os.path.join(sdf_files, f) for f in os.listdir(sdf_files) if f.endswith('.sdf'))
        )

        data_list, collected_specs = [], None
        for idx, sdf_file in enumerate(sdf_file_list):
            supplier = Chem.SDMolSupplier(sdf_file, removeHs=not self.keep_hs, sanitize=self.sanitize)
            mols = [m for m in supplier if m is not None and m.HasProp('type') and m.GetProp('type') in self.input_type]

            if len(mols) < 2:
                print(f"Skipping {sdf_file}: need at least 2 valid molecules with a valid 'type'.")
                continue

            reaction_name = mols[0].GetProp('reaction')
            target_row = self.target_data[self.target_data['rxn'] == reaction_name]
            if target_row.empty:
                print(f"Target data for reaction '{reaction_name}' not found in {sdf_file}.")
                continue

            target = [torch.tensor(target_row[col].values[0], dtype=torch.float) for col in self.target_columns]
            y = torch.stack(target) if len(target) > 1 else target[0]

            tb = TargetBuilder()
            #  Example: assume your CSV has columns 'radius_1', 'radius_2',
            #  'angle', 'dihedral_1', 'dihedral_2'
            row = target_row.iloc[0]                 # pandas Series

            if 'radius_1' in row and 'radius_1' in self.target_columns:
                tb.add_cont("r1_radius", row["radius_1"])
            if 'radius_2' in row and 'radius_2' in self.target_columns:
                tb.add_cont("r2_radius", row["radius_2"])
            if 'angle' in row and 'angle' in self.target_columns:
                tb.add_angle_deg("alpha_angle", row["angle"])
            if 'psi_1_dihedral' in row and 'psi_1_dihedral' in self.target_columns:
                tb.add_sincos_deg("psi_1_dihedral", row["psi_1_dihedral"])
            if 'psi_2_dihedral' in row and 'psi_2_dihedral' in self.target_columns:
                tb.add_sincos_deg("psi_2_dihedral", row["psi_1_dihedral"])
            # tb.add_cont      ("r1_radius",  row["r1_radius"])
            # tb.add_cont      ("r2_radius",  row["r2_radius"])
            # tb.add_angle_deg ("alpha_angle",     row["alpha_angle"])
            # tb.add_sincos_deg("psi_1_dihedral", row["psi_1_dihedral"])
            # tb.add_sincos_deg("psi_2_dihedral", row["psi_2_dihedral"])

            y_raw, specs = tb.build()

            if collected_specs is None:
                collected_specs = specs              # save once
            else:
                assert [s.name for s in specs] == [s.name for s in collected_specs], \
                    "Target spec mismatch across SDFs"
                

            type_dict = {"r1h": "donor", "r2h": "acceptor"}
            data = [
                featurize_molecule(
                    mol_or_smiles=m,
                    target=y_raw,
                    atom_featurizer=self.atom_featurizer,
                    bond_featurizer=self.bond_featurizer,
                    global_featurizer=self.global_featurizer,
                    atom_messages=self.atom_messages,
                    add_hs=self.add_hs,
                    sanitize=self.sanitize,
                    name=str(m.GetProp('reaction')) + "_" + str(m.GetProp('type')) if m.HasProp('type') and m.GetProp('type') else None,
                    comp_type=type_dict.get(m.GetProp('type'), None),
                    mol_properties=json.loads(m.GetProp("mol_properties")) if m.HasProp("mol_properties") else None,
                    electro_map = json.loads(m.GetProp("electro_map")) if m.HasProp("electro_map") and self.use_electro_map else None,
                    idx=idx
                )
                for m in mols
            ]

            data_list.append(data)
        self.target_specs = collected_specs  
        return data_list
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    def inverse_transform_y(self, inplace: bool = True) -> Union[None, List[List[torch.Tensor]]]:
        """
        Inversely transforms all y values using the stored normalizer.

        Args:
            inplace (bool): If True, update each MoleculeData.y in-place. 
                            If False, return a list of unnormalized y values.
        
        Returns:
            None or nested list of torch.Tensors
        """
        if not hasattr(self, 'normalizer') or not self.normalizer_fitted:
            raise RuntimeError("Cannot inverse transform: normalizer not fitted.")


        if inplace:
            for pair in self.data:
                for d in pair:
                    d.y = self.normalizer.inverse_transform(d.y)
            return None

        # -------- Batched inverse transform -------- #
        y_flat = [d.y for pair in self.data for d in pair]
        y_inv_flat = self.normalizer.inverse_transform(torch.stack(y_flat))

        nested = []
        idx = 0
        for pair in self.data:
            n = len(pair)
            nested.append([y_inv_flat[idx + i] for i in range(n)])
            idx += n

        # Set normalizer_fitted to False
        if inplace:
            self.normalizer_fitted = False
        return nested

    def fit_normalizer_from_indices(self, indices: List[int]):
        if not hasattr(self, 'normalizer') or self.normalizer is None:
            raise RuntimeError("No normalizer set in dataset.")
        if self.normalizer_fitted:
            raise RuntimeError("Normalizer already fitted. Use inverse_transform_y to revert.")
        ys = torch.stack([self.data[i][0].y for i in indices])  # assume donor/acceptor y are same
        self.normalizer.fit(ys)

        new_specs, col_ptr = [], 0
        for sp in self.target_specs:
            if sp.kind == "periodic":                    # *scalar* periodic target
                w = self.normalizer.expansion_width   # 2
                new_specs.append(TargetSpec(sp.name, "sincos",
                                            slice(col_ptr, col_ptr+w)))
                col_ptr += w
            else:
                new_specs.append(TargetSpec(sp.name, sp.kind,
                                            slice(col_ptr, col_ptr+1),
                                            mu=self.normalizer.mu[col_ptr],
                                            std=self.normalizer.std[col_ptr]))
                col_ptr += 1

        self.target_specs = new_specs 
        self.normalizer_fitted = True


    def transform_y_from_indices(self, indices: List[int]):
        if not hasattr(self, 'normalizer') or self.normalizer is None:
            raise RuntimeError("No normalizer set in dataset.")
        if not hasattr(self, 'normalizer_fitted') or not self.normalizer_fitted:
            raise RuntimeError("You must fit the normalizer before transforming.")
        for i in indices:
            for d in self.data[i]:
                d.y = self.normalizer.transform(d.y)

    def fit_atom_feature_normalizer_from_indices(self, indices: List[int]):
        if not hasattr(self, 'atom_feature_normalizer') or self.atom_feature_normalizer is None:
            raise RuntimeError("No atom_feature_normalizer set.")
        selected = [self.data[i] for i in indices]
        self.atom_feature_normalizer.fit(selected)
    
    def transform_all_f_atoms(self):
        if not hasattr(self, 'atom_feature_normalizer') or self.atom_feature_normalizer is None:
            return
        for pair in self.data:
            for d in pair:
                d.f_atoms = self.atom_feature_normalizer.transform(d.f_atoms)

    def inverse_transform_all_f_atoms(self):
        if not hasattr(self, 'atom_feature_normalizer') or self.atom_feature_normalizer is None:
            return
        for pair in self.data:
            for d in pair:
                d.f_atoms = self.atom_feature_normalizer.inverse_transform(d.f_atoms)
