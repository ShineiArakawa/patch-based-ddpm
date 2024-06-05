#!/bin/bash

DIVECE_ID=4,5,6,7
CONFIG_FILE="configs/config_patch_divider=4.json"
CHECKPOINT="logs/celeba128/checkpoints/model-last.pt"
OUT_DIR="sampled/celeba128-4x4"
NUM_SAMPLES=16
BATCH_SIZE=4

CUDA_VISIBLE_DEVICES=${DIVECE_ID} accelerate launch -m patch_based_ddpm.sample \
    --config ${CONFIG_FILE} \
    --ckpt ${CHECKPOINT} \
    --out-dir ${OUT_DIR} \
    --n-samples ${NUM_SAMPLES} \
    --batch-size ${BATCH_SIZE}
