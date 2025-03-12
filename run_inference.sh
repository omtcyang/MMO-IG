#!/bin/bash

# 设置第三块 GPU
export CUDA_VISIBLE_DEVICES=6

# 执行推理命令
python inference2.py \
    --json_file data/image_data20_.json \
    --output_dir ./generated_ori_images_6 \
    --weights_path checkpoints/epoch\=40_ori_dior.ckpt \
    --num_samples 6
    
