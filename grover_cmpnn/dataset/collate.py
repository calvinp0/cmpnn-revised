from torch.nn.utils.rnn import pad_sequence
from cmpnn.data.molecule_data import MoleculeDataBatch, MoleculeData
import torch

MASK_TOKEN_ID = 0

def mol_masked_collate(batch):
    batch_graph = MoleculeDataBatch.from_data_list(batch)

    # Prepend dummy row to align with f_atoms (i.e., [0] index is dummy)
    dummy = torch.tensor([-100], dtype=torch.long, device=batch[0].labels.device)

    batch_graph.input_ids = torch.cat([dummy] + [mol.input_ids for mol in batch], dim=0)
    batch_graph.labels = torch.cat([dummy] + [mol.labels for mol in batch], dim=0)

    return batch_graph

