#!/bin/bash
set -e

ENV_NAME="3detr"

echo "=== Creating conda environment: $ENV_NAME ==="
conda create -n $ENV_NAME python=3.6 -y

echo "=== Installing PyTorch 1.9.0 with CUDA 11.1 + pinned MKL ==="
conda run -n $ENV_NAME conda install \
    pytorch==1.9.0 torchvision==0.10.0 \
    cudatoolkit=11.1 \
    mkl==2021.4.0 \
    -c pytorch -c conda-forge -y

echo "=== Installing Python dependencies ==="
conda run -n $ENV_NAME pip install \
    matplotlib \
    opencv-python==4.5.5.64 \
    plyfile \
    "trimesh>=2.35.39,<2.35.40" \
    "networkx>=2.2,<2.3" \
    scipy \
    tensorboard \
    tensorboardX \
    open3d

echo "=== Installing Cython ==="
conda run -n $ENV_NAME conda install cython -y

echo "=== Building pointnet2 CUDA extensions ==="
if [ ! -d "third_party/pointnet2" ]; then
    echo "ERROR: Run this script from the root of the 3detr repo."
    exit 1
fi

conda run -n $ENV_NAME bash -c "cd third_party/pointnet2 && python setup.py install"

echo "=== Compiling Cython box intersection ==="
conda run -n $ENV_NAME bash -c "cd utils && python cython_compile.py build_ext --inplace"

echo "=== Done! Activate with: conda activate $ENV_NAME ==="
