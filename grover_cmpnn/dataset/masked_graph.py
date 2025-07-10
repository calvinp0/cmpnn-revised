import torch
from torch.utils.data import Dataset
import random
from cmpnn.featurizer.utils import featurize_molecule
from collections import defaultdict
from rdkit import Chem
from collections import OrderedDict
def canonicalize(smi):
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol, canonical=True) if mol else None

def deduplicate_smiles(smiles_list, token_ids_list):
    seen = OrderedDict()
    for smi, toks in zip(smiles_list, token_ids_list):
        canon = canonicalize(smi)
        if canon and canon not in seen:
            seen[canon] = (smi, toks)
    return zip(*seen.values())  # returns (deduped_smiles, deduped_token_ids)

class MaskedMoleculeDataset(Dataset):
    def __init__(self, smiles_list, token_ids_list, atom_featurizer, bond_featurizer, add_hs=False, sanitize=True,
                 global_featurizer=None, mask_token_id=0, mask_prob=0.15, k_per_class: int = None,string_dedupe = True, canonicalize_dedupe = False, atom_messages=False):
        assert len(smiles_list) == len(token_ids_list)
        if string_dedupe:
            from collections import OrderedDict
            unique_data = OrderedDict()
            for smi, tok in zip(smiles_list, token_ids_list):
                if smi not in unique_data:
                    unique_data[smi] = tok
            smiles_list = list(unique_data.keys())
            token_ids_list = list(unique_data.values())
        if canonicalize_dedupe:
            smiles_list, token_ids_list = deduplicate_smiles(smiles_list, token_ids_list)

        self.smiles = smiles_list
        self.add_hs = add_hs
        self.k_per_class = k_per_class
        self.sanitize = sanitize
        self.token_ids = token_ids_list
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer
        self.global_featurizer = global_featurizer
        self.mask_token_id = mask_token_id
        self.mask_prob = mask_prob
        self.fixed_smiles = smiles_list
        self.atom_messages = atom_messages

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles = self.fixed_smiles[idx]
        token_ids = self.token_ids[idx]

        mol = featurize_molecule(smiles=smiles, target=None,
                                atom_featurizer=self.atom_featurizer,
                                bond_featurizer=self.bond_featurizer,
                                global_featurizer=self.global_featurizer,
                                add_hs=self.add_hs, sanitize=self.sanitize,
                                atom_messages=self.atom_messages)

        assert len(token_ids) == mol.f_atoms.size(0), (
            f"Mismatch! token_ids: {len(token_ids)}, atoms: {mol.f_atoms.size(0)}"
        )

        # 1. Build index list for each token
        indices_by_token = defaultdict(list)
        for i, tok in enumerate(token_ids):
            indices_by_token[tok].append(i)

        # 2. Sample 15% of each token type
        masked_indices = []
        if self.k_per_class is not None:
            for token, idxs in indices_by_token.items():
                if len(idxs) <= self.k_per_class:
                    masked_indices.extend(idxs)
                else:
                    masked_indices.extend(random.sample(idxs, self.k_per_class))
        else:
            for token, idxs in indices_by_token.items():
                k = max(1, int(self.mask_prob * len(idxs)))
                masked_indices.extend(random.sample(idxs, k))
        # 3. Build input_ids and labels
        input_ids = []
        labels = []
        for i, tok in enumerate(token_ids):
            if i in masked_indices:
                input_ids.append(self.mask_token_id)
                labels.append(tok)  # prediction target
            else:
                input_ids.append(tok)
                labels.append(-100)  # ignored by loss
        mol.input_ids = torch.tensor(input_ids, dtype=torch.long)
        mol.labels = torch.tensor(labels, dtype=torch.long)
        return mol
