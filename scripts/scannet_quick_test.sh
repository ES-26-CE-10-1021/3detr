#!/bin/bash

cd "$(dirname "$0")/.." || exit 1
python main.py \
--dataset_name scannet \
--test_ckpt outputs/scannet_quick/checkpoint_best.pth \
--test_only \
--display_bounding_boxes \
--nqueries 256 \
--dataset_num_workers 8 \
--batchsize_per_gpu 8
