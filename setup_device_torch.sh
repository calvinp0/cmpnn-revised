#!/bin/bash
# Safe to inspect and run with:
# bash setup_device_torch.sh

set -e

echo "Choose backend: [cpu, cuda, rocm]"
read -r backend

if [[ $backend == "cuda" ]]; then
	echo "Choose CUDA version for PyTorch 2.6.0 [11.8, 12.4, 12.6]"
	read -r cuda_ver

	case "$cuda_ver" in
	12.6 | 12.6.0)
		CUDA_TAG="cu126"
		PYG_URL="https://data.pyg.org/whl/torch-2.6.0+cu126.html"
		;;
	12.4 | 12.4.0)
		CUDA_TAG="cu124"
		PYG_URL="https://data.pyg.org/whl/torch-2.6.0+cu124.html"
		;;
	11.8 | 11.8.0)
		CUDA_TAG="cu118"
		PYG_URL="https://data.pyg.org/whl/torch-2.6.0+cu118.html"
		;;
	*)
		echo "Invalid CUDA version: choose one of [11.8, 12.4, 12.6]"
		exit 1
		;;
	esac

	echo "Installing PyTorch (CUDA ${cuda_ver})"
	pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/${CUDA_TAG}
	pip install torch_geometric
	pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f ${PYG_URL}
	pip install lightning

elif [[ $backend == "cpu" ]]; then
	echo "Installing PyTorch (CPU)"
	pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
	pip install torch_geometric
	pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
	pip install lightning

elif [[ $backend == "rocm" ]]; then
	echo "Installing PyTorch (ROCm 6.2.4)"
	pip install torch==2.6.0+rocm6.2.4 torchvision==0.21.0+rocm6.2.4 torchaudio==2.6.0+rocm6.2.4 --index-url https://download.pytorch.org/whl/rocm6.2.4
	pip install lightning
	echo "Downloading PyG ROCm wheels"
	wget https://github.com/Looong01/pyg-rocm-build/releases/download/9/torch-2.6-rocm-6.2.4-py312-linux_x86_64.zip
	unzip torch-2.6-rocm-6.2.4-py312-linux_x86_64.zip
	pip install torch_*.whl
	# Remove downloaded zip file
	rm -v torch-2.6-rocm-6.2.4-py312-linux_x86_64.zip
	# Remove the wheels
	rm -v torch_*.whl

else
	echo "Invalid backend: choose one of [cpu, cuda, rocm]"
	exit 1
fi
