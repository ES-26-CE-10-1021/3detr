#!/usr/bin/env bash
set -eo pipefail

ENV_NAME="3detr"
export TORCH_CUDA_ARCH_LIST="8.0"
# https://developer.nvidia.com/cuda-gpus
# Adjust TORCH_CUDA_ARCH_LIST for your GPU (e.g. 8.6 for RTX 3080, 8.0 for A100)

echo "=== Creating conda environment: ${ENV_NAME} ==="
conda create -n "${ENV_NAME}" python=3.10 -y

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "=== Installing PyTorch 2.5.0 with CUDA 12.4 ==="
conda install pytorch=2.5.0 torchvision=0.20.0 pytorch-cuda=12.4 \
    -c pytorch -c nvidia -y

echo "=== Installing Python dependencies ==="
pip install \
    matplotlib \
    opencv-python \
    plyfile \
    trimesh \
    networkx \
    scipy \
    tensorboard \
    tensorboardX \
    open3d \
    cython \
    numpy

echo "=== Building PointNet++ CUDA extensions ==="
if [ ! -d "third_party/pointnet2" ]; then
    echo "ERROR: Run this script from the root of the 3detr repo."
    exit 1
fi

pip install --no-build-isolation ./third_party/pointnet2

echo "=== Compiling Cython box intersection ==="
(cd utils && python cython_compile.py build_ext --inplace)

echo "=== Done! Activate with: conda activate ${ENV_NAME} ==="
