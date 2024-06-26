#!/bin/bash

DIVECE_ID=4,5,6,7
CONFIG_FILE="configs/config_patch_divider=4.json"
RESTART_STEP=1 # '1' means starting from scratch

CUDA_VISIBLE_DEVICES=${DIVECE_ID} accelerate launch -m patch_based_ddpm.train \
    --config ${CONFIG_FILE} \
    --restart-step ${RESTART_STEP}
