#!/bin/bash
# Usage: sh lps_eval.sh <checkpoint_dir> <checkpoint_step> [robot] [extra eval.py flags]
export MUJOCO_GL=egl

CKPT=$1
STEP=$2
ROBOT=${3:-droid}
shift $(( $# < 3 ? $# : 3 ))

python eval.py \
    --checkpoint_path $CKPT \
    --checkpoint_step $STEP \
    --robot $ROBOT \
    --seed 100 \
    "$@"
