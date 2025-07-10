# file: atom_vocab.py

import pickle
from collections import Counter

class AtomVocab:
    def __init__(self, stoi: dict[str, int]):
        self.stoi = dict(stoi)
        self.itos = {i: s for s, i in self.stoi.items()}

        self.pad_token = '<pad>'
        self.mask_token = '<mask>'
        self.other_token = '<other>'
        self.cls_token = '<cls>' if '<cls>' in self.stoi else None
        
        # Add <mask> and optionally <cls> if not present
        next_id = len(self.stoi)
        for special in [self.mask_token, self.cls_token]:
            if special and special not in self.stoi:
                self.stoi[special] = next_id
                self.itos[next_id] = special
                next_id += 1

        self.pad_token_id = self.stoi[self.pad_token]
        self.mask_token_id = self.stoi[self.mask_token]
        self.other_token_id = self.stoi[self.other_token]
        self.cls_token_id = self.stoi.get(self.cls_token, None)

    def encode(self, atom_types: list[str]) -> list[int]:
        return [self.stoi.get(tok, self.other_token_id) for tok in atom_types]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.itos.get(i, '<unk>') for i in ids]

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.stoi, f)

    @staticmethod
    def load(path: str):
        with open(path, 'rb') as f:
            stoi = pickle.load(f)
        return AtomVocab(stoi)

    def __len__(self):
        return len(self.stoi)

    def __getitem__(self, key):
        return self.stoi[key]

    def get(self, key, default=None):
        return self.stoi.get(key, default)
