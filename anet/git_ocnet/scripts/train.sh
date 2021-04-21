#!/usr/bin/env bash

# train
#CUDA_VISIBLE_DEVICES=0 python train.py --model ocnet \
#    --backbone resnet101 --dataset teeth \
#    --lr 0.0001 --epochs 100 --batch-size 2

#CUDA_VISIBLE_DEVICES=0 python train.py --model ocnet \
#    --backbone resnet152 --dataset teeth \
#    --lr 0.0001 --epochs 100 --batch-size 1


CUDA_VISIBLE_DEVICES=0 python train.py --model ocnet \
    --backbone resnet101 --dataset teeth \
    --lr 0.0001 --epochs 100 --batch-size 2

#CUDA_VISIBLE_DEVICES=0 python train.py --model ocnet \
#    --backbone resnet50 --dataset teeth \
#    --lr 0.0001 --epochs 100 --batch-size 2

#CUDA_VISIBLE_DEVICES=0 python train.py --model ocnet \
#   --backbone resnet50 --dataset teeth \
#   --lr 0.0001 --epochs 100 --batch-size 1


# CUDA_VISIBLE_DEVICES=0 python train.py --model psp \
#     --backbone resnet101 --dataset teeth \
#     --lr 0.0001 --epochs 100 --batch-size 2