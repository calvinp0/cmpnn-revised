# CMPNN-Revised

A modular, extensible, and Lightning-powered revision of [SY575/CMPNN](https://github.com/SY575/CMPNN) — based on the IJCAI 2020 paper:  
**[Communicative Representation Learning on Attributed Molecular Graphs](https://www.ijcai.org/Proceedings/2020/0392.pdf)**

## 🚀 Overview

This repository revises and modernizes the original Communicative Message Passing Neural Network (CMPNN) implementation to improve:

- **Extensibility** – Modular design for easy plug-and-play experimentation.
- **Readability** – Cleaner abstractions and more maintainable code.
- **Trainability** – Integrated [PyTorch Lightning](https://www.pytorchlightning.ai/) for scalable training loops and device handling.
- **Completeness** – Implements options *mentioned in the paper* but missing from the original repo.

## ✅ What's New

### 🧠 Architecture
- **Modular CMPNNEncoder** with configurable:
  - `comm_mode`: `'add'`, `'mlp'`, `'gru'`, `'ip'`
  - `booster`: `'sum'`, `'mean'`, `'sum_max'`, `'attention'`
- **Aggregation strategies** extracted into their own classes: easy to swap in `mean`, `sum`, `norm`, etc.
- Shared encoder or separate encoders for multi-molecule models.

### 🔁 Multi-Molecule Learning
- Support for **pairwise molecule encoding**:
  - Use shared CMPNN encoders or independent ones.
  - Encode donor/acceptor, ligand/protein, or other pairwise molecular structures.

### 🔧 Optimization & Infrastructure
- Refactored with **`torch_geometric`**-compatible `Data` and `Batch` objects.
- Optional **global features** using:
  - Morgan fingerprints
  - RDKit 2D descriptors
  - Normalized RDKit features
  - Charge-based features
- Extended featurizers and batching logic with robust testing.

## 📦 Structure

```bash
cmpnn_revised/
├── models/               # CMPNN core architecture
├── data/                 # Data objects & batching logic
├── featurizer/           # Atom/Bond/Global featurizers
├── lightning/            # PyTorch Lightning modules
├── scripts/              # Training, evaluation, etc.
├── tests/                # Pytest unit tests
```
## 📚 Reference

Communicative Representation Learning on Attributed Molecular Graphs
Shengchao Liu, Xuanang Li, Xuanjing Huang, Jian Tang
IJCAI 2020 — [PDF](https://www.ijcai.org/Proceedings/2020/0392.pdf)

## 🛠 Installation

Create a Conda environment and install dependencies:
```bash
conda env create -f environment.yml
conda activate cmpnn_env
bash setup_device_torch.sh
```

## 🔬 Example Usage

Coming soon. Check out scripts/ and tests/ for sample training workflows and test coverage.

## 💡 Acknowledgements

- Original inspiration and code: [SY575/CMPNN](https://github.com/SY575/CMPNN)
- Built using RDKit, PyTorch Geometric, and PyTorch Lightning
