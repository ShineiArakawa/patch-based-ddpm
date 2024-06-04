#!/bin/bash

DIVECE_ID=0,1,2,3
CONFIG_FILE="configs/config_patch_divider=4.json"
CHECKPOINT="log/celeba128/checkpoints/model-4000.pt"
OUT_DIR="sampled/celeba128_4x4"
NUM_SAMPLES=32
BATCH_SIZE=4

CUDA_VISIBLE_DEVICES=${DIVECE_ID} accelerate launch -m patch_based_ddpm.sample \
    --config ${CONFIG_FILE} \
    --ckpt ${CHECKPOINT} \
    --out-dir ${OUT_DIR} \
    --n-samples ${NUM_SAMPLES} \
    --batch-size ${BATCH_SIZE}
