#!/bin/bash
# Safe to inspect and run with:
# bash setup_device_torch.sh

set -e

echo "Choose backend: [cpu, cuda, rocm]"
read -r backend

if [[ "$backend" == "cuda" ]]; then
    echo "Installing PyTorch (CUDA 12.6)"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    pip install torch_geometric
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu126.html
    pip install lightning

elif [[ "$backend" == "cpu" ]]; then
    echo "Installing PyTorch (CPU)"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install torch_geometric
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
    pip install lightning

elif [[ "$backend" == "rocm" ]]; then
    echo "Installing PyTorch (ROCm 6.2.4)"
    pip install torch==2.6.0+rocm6.2.4 torchvision==0.21.0+rocm6.2.4 torchaudio==2.6.0+rocm6.2.4 --index-url https://download.pytorch.org/whl/rocm6.2.4
    pip install lightning
    echo "Downloading PyG ROCm wheels"
    wget https://github.com/Looong01/pyg-rocm-build/releases/download/9/torch-2.6-rocm-6.2.4-py312-linux_x86_64.zip
    unzip torch-2.6-rocm-6.2.4-py312-linux_x86_64.zip
    pip install torch_*.whl

else
    echo "Invalid backend: choose one of [cpu, cuda, rocm]"
    exit 1
fi
