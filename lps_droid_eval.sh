#!/bin/bash
MUJOCO_GL=egl
python droid_eval.py \
    --checkpoint_path $1 \
    --checkpoint_step $2 \
    --seed 100
