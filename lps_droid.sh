#!/bin/bash
# Usage: sh lps_droid.sh <task_name> <droid_dataset_dir> <seed> [extra main.py flags]
# e.g. append --droid_use_failure to also train on the failure/ trajectories.

export MUJOCO_GL="egl"

TASK=$1
DATA_DIR=$2
SEED=$3
shift 3

python main.py \
    --agent "agents/lps.py" \
    --project "LPS_DROID" \
    --run_group "LPS:DROID:$TASK" \
    --droid_dataset_dir "$DATA_DIR" \
    --task_name $TASK \
    --task_num 0 \
    --env_name "$TASK" \
    --horizon_length 5 \
    --agent.lr 3e-4 \
    --save_interval 5000 \
    --offline_steps 10000 \
    --save_dir "exp_droid/" \
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
