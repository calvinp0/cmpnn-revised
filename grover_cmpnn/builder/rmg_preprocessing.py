# Run inside rmg_env
try:
    from grover_cmpnn.builder.rmg_vocab_builder import build_atom_vocab, smiles_to_atom_ids
except ImportError:
    import sys
    sys.path.append('/home/calvin/code/grover_cmpnn/grover_cmpnn/builder')
    from rmg_vocab_builder import build_atom_vocab, smiles_to_atom_ids
import pandas as pd
from rdkit import Chem


molecules = Chem.SupplierFromFilename("/home/calvin/code/cmpnn_revised/molnet_data/qm9_data/raw/gdb9.sdf", sanitize=False)
smiles_list = [Chem.MolToSmiles(mol) for mol in molecules]

# 1. Build vocab from corpus (if not done already)
vocab, success_smiles = build_atom_vocab(smiles_list, min_freq=10, save_path="atom_vocab.pkl")

# 2. Convert all SMILES to atom ID sequences
all_atom_ids = [smiles_to_atom_ids(smi, vocab) for smi in smiles_list if smiles_to_atom_ids(smi, vocab) is not None]

# 3. Save
import pickle
with open("token_ids.pkl", "wb") as f:
    pickle.dump(all_atom_ids, f)


# 4. Save the SMILES strings to a CSV file
df = pd.DataFrame({"smiles": success_smiles})
df.to_csv("smiles.csv", index=False)
