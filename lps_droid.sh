#!/bin/bash

export MUJOCO_GL="egl"

python main.py \
    --agent "agents/lps.py" \
    --project "LPS_DROID" \
    --run_group "LPS:DROID:$1" \
    --droid_dataset_dir "$2" \
    --task_name $1 \
    --task_num 0 \
    --env_name "$1" \
    --horizon_length 5 \
    --agent.lr 3e-4 \
    --eval_interval 10000 \
    --save_interval 5000 \
    --offline_steps 10000 \
    --save_dir "exp_droid/" \
    --agent.num_critic 2 \
    --agent.alpha 1.0 \
    --agent.extract_method "ddpg" \
    --agent.mf_method "jit_mf" \
    --seed $3 \
    --agent.encoder "impala" \
    --p_aug=0.5 \
    --agent.use_DiT \
    --agent.size_DiT "small" \
    --log_interval 100 \
    --agent.critic_hidden_dims "(512, 512, 512, 512)" \
    --agent.latent_actor_hidden_dims "(512, 512)" \
    # --droid_use_failure
