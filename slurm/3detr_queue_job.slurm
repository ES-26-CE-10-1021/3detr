#!/bin/bash
#SBATCH --job-name=3detr
#SBATCH --output=logs/3detr_%j.log
#SBATCH --error=logs/3detr_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:2
#SBATCH --time=7-00:00:00
#SBATCH --mem=96G

# Activate conda env
source ~/miniforge3/bin/activate 3detr

# Run script
srun bash ~/3D-Perception/3detr/scripts/scannet_quick_modified.sh
