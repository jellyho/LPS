#!/bin/bash
# Usage: sh lps_lerobot.sh <dataset_name> <lerobot_dataset_dir> <seed> [extra main.py flags]

export MUJOCO_GL="egl"

NAME=$1
DATA_DIR=$2
SEED=$3
shift 3

python main.py \
    --agent "agents/lps.py" \
    --project "LPS_LEROBOT" \
    --run_group "LPS:LEROBOT:$NAME" \
    --lerobot_dataset_dir "$DATA_DIR" \
    --task_name $NAME \
    --task_num 0 \
    --env_name "$NAME" \
    --horizon_length 5 \
    --agent.lr 3e-4 \
    --save_interval 5000 \
    --offline_steps 10000 \
    --save_dir "exp_lerobot/" \
    --agent.num_critic 2 \
    --agent.alpha 1.0 \
    --seed $SEED \
    --agent.encoder "impala" \
    --p_aug=0.5 \
    --agent.use_DiT \
    --agent.size_DiT "small" \
    --log_interval 100 \
    --agent.critic_hidden_dims "(512, 512, 512, 512)" \
    --agent.latent_actor_hidden_dims "(512, 512)" \
    --agent.latent_actor_layer_norm=True \
    --agent.latent_entropy_per_dim=True \
    "$@"
