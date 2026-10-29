import os
from datetime import datetime

import absl.flags as flags
import ml_collections
import wandb

def get_exp_name(seed):
    """Return the experiment name."""
    exp_name = ''
    exp_name += f'sd{seed:03d}'
    if 'SLURM_JOB_ID' in os.environ:
        exp_name += f's_{os.environ["SLURM_JOB_ID"]}.'
    if 'SLURM_PROCID' in os.environ:
        exp_name += f'{os.environ["SLURM_PROCID"]}.'
    if 'SLURM_ARRAY_JOB_ID' in os.environ:
        exp_name += f'{os.environ["SLURM_ARRAY_JOB_ID"]}.'
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        exp_name += f'{os.environ["SLURM_ARRAY_TASK_ID"]}.'
    exp_name += f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    return exp_name


def get_flag_dict():
    """Return the dictionary of flags."""
    flag_dict = {k: getattr(flags.FLAGS, k) for k in flags.FLAGS if '.' not in k}
    for k in flag_dict:
        if isinstance(flag_dict[k], ml_collections.ConfigDict):
            flag_dict[k] = flag_dict[k].to_dict()
    return flag_dict


def setup_wandb(
    entity=None,
    project='project',
    group=None,
    name=None,
    mode='online',
):
    """Set up Weights & Biases for logging."""
    tags = [group] if group is not None else None

    init_kwargs = dict(
        config=get_flag_dict(),
        project=project,
        entity=entity,
        tags=tags,
        group=group,
        name=name,
        settings=wandb.Settings(_disable_stats=False),
        mode=mode,
    )

    run = wandb.init(**init_kwargs)

    # assume a flat structure
    run.save('*.py')
    run.save('**/*.py')

    return run


