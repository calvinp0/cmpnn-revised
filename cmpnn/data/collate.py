from cmpnn.data.molecule_data import MoleculeDataBatch, MultiMoleculeDataBatch


def mol_collate_fn(batch):
    """
    Collate function for a batch of MoleculeData objects.
    This function is used to combine multiple MoleculeData objects into a single batch.
    """
    # Check if the batch is empty
    if len(batch) == 0:
        return None

    # Use the from_data_list method to create a batch from the list of MoleculeData objects
    return MoleculeDataBatch.from_data_list(batch)


def multimol_collate_fn(batch):
    """
    Collate function for a batch of MultiMoleculeData objects.
    This function is used to combine multiple MultiMoleculeData objects into a single batch.
    """
    # Check if the batch is empty
    if len(batch) == 0:
        return None
    return MultiMoleculeDataBatch.from_data_list(batch)
