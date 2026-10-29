#!/bin/bash
# Usage: sh lps_ogbench.sh <agent_name> <task_name> <task_num> <alpha> <seed> [extra main.py flags]

export MUJOCO_GL="egl"

AGENT=$1
TASK=$2
TASK_NUM=$3
ALPHA=$4
SEED=$5
shift 5

python main.py \
    --agent "agents/$AGENT.py" \
    --project "LPS" \
    --run_group "LPS:$AGENT:$TASK" \
    --task_name $TASK \
    --task_num $TASK_NUM \
    --env_name "$TASK-singletask-task$TASK_NUM-v0" \
    --horizon_length 5 \
    --agent.lr 3e-4 \
    --eval_interval 100000 \
    --eval_episodes 50 \
    --video_episodes 10 \
    --offline_steps 1000000 \
    --save_dir "exp/" \
    --agent.num_critic 2 \
    --agent.alpha $ALPHA \
    --seed $SEED \
    "$@"
