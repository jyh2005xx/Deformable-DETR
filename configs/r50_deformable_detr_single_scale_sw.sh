#!/usr/bin/env bash

set -x
conda activate ddetr
EXP_DIR=exps/r50_deformable_detr_single_scale
PY_ARGS=${@:1}

python -u main.py \
    --num_feature_levels 1 \
    --swas \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS}
