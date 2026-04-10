#!/bin/bash

export MUJOCO_GL="egl"

python main.py \
    --agent "agents/$1.py" \
    --project "LPS" \
    --run_group "LPS:$1:$2" \
    --task_name $2 \
    --task_num $3 \
    --env_name "$2-singletask-task$3-v0" \
    --horizon_length 5 \
    --agent.lr 3e-4 \
    --eval_interval 100000 \
    --eval_episodes 50 \
    --video_episodes 10 \
    --offline_steps 1000000 \
    --save_dir "exp/" \
    --agent.num_critic 2 \
    --agent.alpha $4 \
    --seed $5 \
