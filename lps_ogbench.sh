#!/bin/bash

export MUJOCO_GL="egl"

python main.py \
    --agent "agents/lps.py" \
    --project "LPS" \
    --run_group "LPS:$1:$2" \
    --task_name $1 \
    --task_num $2 \
    --env_name "$1-singletask-task$2-v0" \
    --horizon_length 5 \
    --agent.lr 3e-4 \
    --eval_interval 100000 \
    --eval_episodes 50 \
    --video_episodes 10 \
    --agent.extract_method "ddpg" \
    --offline_steps 1000000 \
    --save_dir "exp/" \
    --agent.num_critic 2 \
    --agent.latent_dist "sphere" \
    --agent.alpha 1.0 \
    --seed $3 \
    --agent.latent_actor_hidden_dims "(256,256)" \
    --agent.critic_agg "min" \
    --agent.mf_method "jit_mf"
