from torch.utils.data import Dataset
import torch
import random
import pickle

MASK_TOKEN_ID = 0

def mask_atom_ids(atom_ids: list[int], mask_token_id: int = MASK_TOKEN_ID, mask_prob: float = 0.15):
    """
    Apply BERT-style random masking to a list of atom vocab IDs.
    Returns:
        - masked_ids: input with some tokens replaced by mask_token_id
        - labels: original token IDs where masked, -100 elsewhere (so ignored in loss)
    """
    masked_ids = []
    labels = []

    for token_id in atom_ids:
        if random.random() < mask_prob:
            masked_ids.append(mask_token_id)
            labels.append(token_id)
        else:
            masked_ids.append(token_id)
            labels.append(-100)  # ignored in loss

    return masked_ids, labels

class MaskedAtomDataset(Dataset):
    def __init__(self, token_sequences, vocab, mask_prob=0.15, add_cls_token=True):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.mask_prob = mask_prob
        self.add_cls_token = add_cls_token
        
        self.pad_token_id = vocab['<pad>']
        self.mask_token_id = vocab['<mask>']
        self.cls_token_id = vocab.get('<cls>', None)

        # Load token sequences
        if isinstance(token_sequences, str) and token_sequences.endswith('.pkl'):
            with open(token_sequences, 'rb') as f:
                self.data = pickle.load(f)
        else:
            self.data = token_sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        token_ids = self.data[idx]
        input_ids = []
        labels = []

        for token_id in token_ids:
            if random.random() < self.mask_prob:
                input_ids.append(self.mask_token_id)
                labels.append(token_id)
            else:
                input_ids.append(token_id)
                labels.append(-100)

        if self.add_cls_token:
            input_ids = [self.cls_token_id] + input_ids
            labels = [-100] + labels

        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels)
        }
