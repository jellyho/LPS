import os
import glob
import h5py
import numpy as np
from tqdm import tqdm

def load_droid_dataset(
    root_dir, 
    include_failure=True, 
    discount=0.99, 
    compact_dataset=False,
    sparse=False,
    horizon_length=5,
):
    """
    Load Franka HDF5 dataset from root_dir with 'success' and 'failure' subfolders.
    
    Args:
        root_dir: Root directory containing 'success' and 'failure' folders.
        include_failure: Whether to include trajectories from the 'failure' folder.
        discount: Discount factor (gamma) used for calculating failure terminal reward.
        compact_dataset: If True, does not generate 'next_observations'.
        
    Returns:
        A dictionary compatible with the JAX Dataset class.
    """
    
    # 1. Collect all file paths
    success_dir = os.path.join(root_dir, 'success')
    failure_dir = os.path.join(root_dir, 'failure')
    
    # Find all .h5 files
    success_paths = glob.glob(os.path.join(success_dir, '*.h5'))
    failure_paths = glob.glob(os.path.join(failure_dir, '*.h5')) if include_failure else []
    
    all_paths = success_paths + failure_paths
    
    # Sort to ensure reproducibility
    all_paths.sort()
    
    # Store lists to concatenate later
    data_lists = {
        'observations': {'state': [], 'images': []},
        'actions': [],
        'terminals': [],
        'rewards': [],
        'masks': [] # Useful for bootstrapping logic
    }
    
    print(f"Loading {len(success_paths)} success and {len(failure_paths)} failure trajectories...")

    for filepath in tqdm(all_paths, desc="Reading HDF5"):
        try:
            is_success = 'success' in filepath
            
            with h5py.File(filepath, 'r') as f:
                # --- 2. Action Parsing ---
                # action/cartesian_velocity + action/gripper_position
                cart_vel = f['action/cartesian_velocity'][:]
                grip_pos = f['action/gripper_position'][:]
                # Ensure dimensions match (T, D)
                if len(grip_pos.shape) == 1:
                    grip_pos = grip_pos[:, None]
                actions = np.concatenate([cart_vel, grip_pos], axis=-1)
                
                # --- 3. Proprioception Parsing ---
                # observation/robot_state/cartesian_position + observation/robot_state/gripper_position
                robot_cart_pos = f['observation/robot_state/cartesian_position'][:]
                robot_grip_pos = f['observation/robot_state/gripper_position'][:]
                if len(robot_grip_pos.shape) == 1:
                    robot_grip_pos = robot_grip_pos[:, None]
                state = np.concatenate([robot_cart_pos, robot_grip_pos], axis=-1)
                
                # --- 4. Image Parsing ---
                # Find keys matching 'observation/image/*_left'
                img_grp = f['observation/image']
                # Filter keys ending with '_left'
                cam_keys = sorted([k for k in img_grp.keys() if k.endswith('_left')])
                
                if len(cam_keys) != 2:
                    print(f"Warning: Expected 2 cameras, found {len(cam_keys)} in {filepath}. Skipping.")
                    continue
                
                # Load images: resulting shape (T, H, W, C) per camera # default channel is 4, so reduce it!
                imgs_cam1 = img_grp[cam_keys[0]][:][..., :3][..., ::-1] 
                imgs_cam2 = img_grp[cam_keys[1]][:][..., :3][..., ::-1]
                
                # Stack cameras along channel or keeping them separate depends on your encoder.
                # Here, let's stack them along the channel dimension for JAX to handle as one big tensor 
                # OR keep them separate if you use a dict.
                # Given the OGBench code uses tree_map, let's concat channel-wise: (T, H, W, 6)
                # Assuming images are RGB (3 channels)
                images = np.concatenate([imgs_cam1, imgs_cam2], axis=-1)

                traj_len = len(actions)
                
                # --- 5. Reward & Terminal Logic ---
                # Basic Step Reward: -1 per step
                if sparse:
                    rewards = np.full((traj_len,), 0.0, dtype=np.float32)
                else:
                    rewards = np.full((traj_len,), -1.0, dtype=np.float32)
                
                terminals = np.zeros((traj_len,), dtype=np.float32)
                masks = np.ones((traj_len,), dtype=np.float32)
                
                # Terminal Handling (Last H timesteps are all terminal)
                terminals[-1] = 1.0
                masks[-horizon_length:] = 0.0 # Standard RL: do not bootstrap at terminal state
                
                if is_success:
                    # Success: Give 0 reward at the last step (0.5s) (Goal reached)
                    if sparse:
                        rewards[-horizon_length] = 1.0
                    else:
                        rewards[-horizon_length] = 0.0
                    
                else:
                    # Failure: Apply user specific logic
                    if sparse:
                        rewards[-horizon_length] = 0.0
                    else:
                        rewards[-horizon_length] = -100
                    
                
                # Append to lists
                data_lists['observations']['state'].append(state)
                data_lists['observations']['images'].append(images)
                data_lists['actions'].append(actions)
                data_lists['terminals'].append(terminals)
                data_lists['rewards'].append(rewards)
                data_lists['masks'].append(masks)
                
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue

    # 6. Concatenate all trajectories into single arrays
    dataset = {}
    
    # Handle Observations (Nested Dictionary)
    dataset['observations'] = {
        'state': np.concatenate(data_lists['observations']['state'], axis=0).astype(np.float32),
        'image': np.concatenate(data_lists['observations']['images'], axis=0).astype(np.uint8) 
    }
    
    dataset['actions'] = np.concatenate(data_lists['actions'], axis=0).astype(np.float32)
    dataset['terminals'] = np.concatenate(data_lists['terminals'], axis=0).astype(np.float32)
    dataset['rewards'] = np.concatenate(data_lists['rewards'], axis=0).astype(np.float32)
    dataset['masks'] = np.concatenate(data_lists['masks'], axis=0).astype(np.float32)

    # 7. Post-processing (Next Observations logic from original OGBench code)
    if compact_dataset:
        # Compact: Just add valid mask
        dataset['valids'] = 1.0 - dataset['terminals']
        new_terminals = np.concatenate([dataset['terminals'][1:], [1.0]])
        dataset['terminals'] = np.minimum(dataset['terminals'] + new_terminals, 1.0).astype(np.float32)
    else:
        # Regular: Compute next_observations by shifting
        ob_mask = (1.0 - dataset['terminals']).astype(bool)
        
        # Shift Nested Observations
        dataset['next_observations'] = {
            'state': dataset['observations']['state'][1:],
            'image': dataset['observations']['image'][1:] # Be careful with memory here!
        }
        
        # Filter current observations/actions with mask
        # Note: Since next_obs is shifted, we lose the last element, 
        # but the standard logic requires careful masking.
        
        # Re-implementing OGBench logic strictly for nested dicts:
        def apply_shift_and_mask(obs_dict, mask):
            new_obs = {}
            new_next_obs = {}
            next_ob_mask = np.concatenate([[False], mask[:-1]]) # Valid 'next' indices
            
            for k, v in obs_dict.items():
                # dataset['observations'] becomes masked
                new_obs[k] = v[mask]
                # dataset['next_observations'] picks items where previous was valid
                new_next_obs[k] = v[next_ob_mask] 
                
            return new_obs, new_next_obs
        
        next_ob_mask = np.concatenate([[False], ob_mask[:-1]])
        
        # Process nested observations
        dataset['next_observations'] = {}
        for key in dataset['observations']:
            full_arr = dataset['observations'][key]
            dataset['next_observations'][key] = full_arr[next_ob_mask]
            dataset['observations'][key] = full_arr[ob_mask]
            
        dataset['actions'] = dataset['actions'][ob_mask]
        dataset['rewards'] = dataset['rewards'][ob_mask]
        dataset['masks'] = dataset['masks'][ob_mask]
        
        # Fix terminals
        new_terminals = np.concatenate([dataset['terminals'][1:], [1.0]])
        dataset['terminals'] = new_terminals[ob_mask].astype(np.float32)

    print("Dataset creation complete!")
    print(f"Total transitions: {len(dataset['actions'])}")
    
    return dataset