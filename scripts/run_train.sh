#!/bin/bash

DIVECE_ID=0,1
CONFIG_FILE="configs/config_patch_divider=4.json"

CUDA_VISIBLE_DEVICES=${DIVECE_ID} accelerate launch -m patch_based_ddpm.train \
    --config ${CONFIG_FILE}
