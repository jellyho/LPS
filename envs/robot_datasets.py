import os
import glob
import h5py
import numpy as np
from tqdm import tqdm

def load_droid_dataset(
    root_dir, 
    include_failure=False,
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
        discount: Sets the failure penalty, -1 / (1 - discount); see ``outcome_reward``.
        compact_dataset: If True, does not generate 'next_observations'.
        sparse: 0 per step and +1 on success, instead of -1 per step, 0 at the end of a success and
            the failure penalty at the end of a failure.
        horizon_length: Action chunk length; the last chunk of an episode is not bootstrapped.
        
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
                
                # (T, H, W, 4) BGRA per camera: drop the alpha and reverse to RGB.
                imgs_cam1 = img_grp[cam_keys[0]][:][..., :3][..., ::-1]
                imgs_cam2 = img_grp[cam_keys[1]][:][..., :3][..., ::-1]

                # Both views stacked on the channel axis: (T, H, W, 6).
                images = np.concatenate([imgs_cam1, imgs_cam2], axis=-1)

                traj_len = len(actions)
                
                # --- 5. Reward & Terminal Logic ---
                rewards = np.full((traj_len,), outcome_reward(None, sparse, discount), dtype=np.float32)
                
                terminals = np.zeros((traj_len,), dtype=np.float32)
                masks = np.ones((traj_len,), dtype=np.float32)
                
                # The episode ends at its last step; the last chunk is not bootstrapped.
                terminals[-1] = 1.0
                masks[-horizon_length:] = 0.0 # Standard RL: do not bootstrap at terminal state
                
                # The outcome reward lands on the last step a chunk can start from.
                rewards[-horizon_length] = outcome_reward(
                    'success' if is_success else 'failure', sparse, discount)
                    
                
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
        # Regular: next_observations are the observations shifted by one, as in OGBench.
        ob_mask = (1.0 - dataset['terminals']).astype(bool)
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
        
        # Shift terminals to match the shifted observations.
        new_terminals = np.concatenate([dataset['terminals'][1:], [1.0]])
        dataset['terminals'] = new_terminals[ob_mask].astype(np.float32)

    print("Dataset creation complete!")
    print(f"Total transitions: {len(dataset['actions'])}")
    
    dataset['_meta'] = dict(
        camera_keys=list(DROID_CAMERA_ORDER),
        image_shape=list(dataset['observations']['image'].shape[-3:]),
    )
    return dataset

# Serial order: 14013996 (wrist) before 36776608 (exterior), as load_droid_dataset stacks them.
DROID_CAMERA_ORDER = ('observation.images.wrist', 'observation.images.exterior')


def _decode_video_segments(video_path, segments, image_size=None):
    """Decode several episodes out of one aggregated LeRobot mp4 in a single pass.

    LeRobot v3 packs many episodes into one video file, so we stream the file once and keep only the
    frames each episode asks for. ``segments`` is a list of ``(start_frame, length)``; the return
    value is a list of ``(T, H, W, 3)`` uint8 RGB arrays in the same order. Decoding goes through
    imageio's bundled ffmpeg, which reads both the AV1 that LeRobot writes by default and h264.
    """
    import imageio.v2 as imageio

    out = [None] * len(segments)
    last_needed = max(start + length for start, length in segments)
    reader = imageio.get_reader(str(video_path), 'ffmpeg')
    try:
        buffers = [[] for _ in segments]
        for idx, frame in enumerate(reader):
            if idx >= last_needed:
                break
            for i, (start, length) in enumerate(segments):
                if start <= idx < start + length:
                    if image_size is not None and frame.shape[:2] != tuple(image_size):
                        import cv2
                        frame = cv2.resize(
                            frame, (image_size[1], image_size[0]), interpolation=cv2.INTER_AREA
                        )
                    buffers[i].append(frame)
    finally:
        reader.close()

    for i, (start, length) in enumerate(segments):
        if len(buffers[i]) != length:
            raise RuntimeError(
                f'{video_path}: frames {start}..{start + length} gave {len(buffers[i])} frames'
            )
        out[i] = np.stack(buffers[i])
    return out


def _decode_frames(cells, image_size=None):
    """Decode a column of LeRobot image cells -- ``{'bytes': ...}`` PNG or JPEG -- into RGB."""
    import cv2

    out = []
    for cell in cells:
        buf = cell['bytes'] if isinstance(cell, dict) else cell
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)  # BGR
        if image_size is not None and img.shape[:2] != tuple(image_size):
            img = cv2.resize(img, (image_size[1], image_size[0]), interpolation=cv2.INTER_AREA)
        out.append(img[..., ::-1])
    return np.stack(out)


def outcome_reward(outcome, sparse=False, discount=0.99):
    """The robot reward contract, shared by the offline loaders and the online environments.

    ``outcome`` is None while the episode runs, then 'success' or 'failure' on its last step.
    -1 per step, 0 at the end of a success and -1 / (1 - discount) at the end of a failure: the value of
    never succeeding, so ending an episode by failing is never worth more than going on. Under
    ``sparse``, 0 per step and +1 at the end of a success.
    """
    if outcome == 'success':
        return 1.0 if sparse else 0.0
    if outcome == 'failure' and not sparse:
        return -1.0 / (1.0 - discount)
    return 0.0 if sparse else -1.0


def assemble_dataset(kept, images, camera_keys, sparse=False, horizon_length=5,
                     compact_dataset=False, discount=0.99):
    """Reward, terminal and next-observation logic, shared by every robot loader.

    Args:
        kept: one dict per episode with ``state``, ``actions``, ``traj_len`` and ``is_success``.
        images: for each episode, one ``(T, H, W, 3)`` array per camera, in ``camera_keys`` order,
            or None for a dataset observed through its state alone.

    Rewards follow ``outcome_reward``. The outcome reward lands on the last step a chunk of
    ``horizon_length`` can start from, and the last chunk is where the episode ends.
    """
    data_lists = {
        'observations': {'state': [], 'images': []},
        'actions': [], 'terminals': [], 'rewards': [], 'masks': [],
    }
    for ep, per_cam in zip(kept, images if images is not None else [None] * len(kept)):
        traj_len = ep['traj_len']

        rewards = np.full((traj_len,), outcome_reward(None, sparse, discount), dtype=np.float32)

        terminals = np.zeros((traj_len,), dtype=np.float32)
        masks = np.ones((traj_len,), dtype=np.float32)
        terminals[-1] = 1.0
        masks[-horizon_length:] = 0.0

        rewards[-horizon_length] = outcome_reward('success' if ep['is_success'] else 'failure', sparse, discount)

        data_lists['observations']['state'].append(ep['state'])
        if per_cam is not None:
            data_lists['observations']['images'].append(np.concatenate(per_cam, axis=-1))
        data_lists['actions'].append(ep['actions'])
        data_lists['terminals'].append(terminals)
        data_lists['rewards'].append(rewards)
        data_lists['masks'].append(masks)

    if not data_lists['actions']:
        raise ValueError('no usable episodes')

    observations = {
        'state': np.concatenate(data_lists['observations']['state'], axis=0).astype(np.float32)}
    if images is not None:
        observations['image'] = np.concatenate(
            data_lists['observations']['images'], axis=0).astype(np.uint8)
    dataset = {
        'observations': observations,
        'actions': np.concatenate(data_lists['actions'], axis=0).astype(np.float32),
        'terminals': np.concatenate(data_lists['terminals'], axis=0).astype(np.float32),
        'rewards': np.concatenate(data_lists['rewards'], axis=0).astype(np.float32),
        'masks': np.concatenate(data_lists['masks'], axis=0).astype(np.float32),
    }

    if compact_dataset:
        dataset['valids'] = 1.0 - dataset['terminals']
        new_terminals = np.concatenate([dataset['terminals'][1:], [1.0]])
        dataset['terminals'] = np.minimum(dataset['terminals'] + new_terminals, 1.0).astype(np.float32)
    else:
        ob_mask = (1.0 - dataset['terminals']).astype(bool)
        next_ob_mask = np.concatenate([[False], ob_mask[:-1]])

        dataset['next_observations'] = {}
        for key in dataset['observations']:
            full_arr = dataset['observations'][key]
            dataset['next_observations'][key] = full_arr[next_ob_mask]
            dataset['observations'][key] = full_arr[ob_mask]

        dataset['actions'] = dataset['actions'][ob_mask]
        dataset['rewards'] = dataset['rewards'][ob_mask]
        dataset['masks'] = dataset['masks'][ob_mask]

        new_terminals = np.concatenate([dataset['terminals'][1:], [1.0]])
        dataset['terminals'] = new_terminals[ob_mask].astype(np.float32)

    print('Dataset creation complete!')
    print(f"Total transitions: {len(dataset['actions'])}")

    if images is not None:
        dataset['_meta'] = dict(
            camera_keys=list(camera_keys),
            image_shape=list(dataset['observations']['image'].shape[-3:]),
        )
    return dataset

def load_lerobot_dataset(
    root_dir,
    camera_keys=None,
    state_key='observation.state',
    action_key='action',
    reward_key='next.reward',
    discount=0.99,
    compact_dataset=False,
    sparse=False,
    horizon_length=5,
    image_size=None,
    max_episodes=None,
    reverse_channels=None,
    task=None,
):
    """Load a LeRobot (v3.0) dataset into the same dictionary ``load_droid_dataset`` returns.

    Same principle as the DROID path: proprioception and the two camera views are stacked into one
    observation, the reward is sparse, and the last ``horizon_length`` steps are not bootstrapped.
    The difference is only where the bytes come from -- parquet plus aggregated mp4 instead of HDF5
    -- so every agent in ``agents/`` trains on it unchanged.

    Args:
        root_dir: Dataset root (the directory holding ``meta/``, ``data/`` and ``videos/``).
        camera_keys: Video keys to use, in order; defaults to every video key in ``meta/info.json``.
        reward_key: Column holding the per-step reward. If the dataset has no such column, every
            episode is treated as a successful demonstration (the usual case for a LeRobot dataset
            recorded by teleoperation).
        image_size: ``(H, W)`` to resize frames to, or None to keep the recorded size.
        max_episodes: Load at most this many episodes (for smoke tests).
        task: Keep only the episodes whose language instruction matches, for a dataset that holds
            several tasks. Multi-task training is out of scope.
        reverse_channels: Reverse each camera's channels. The default reads ``channel_order`` from
            ``meta/info.json`` and reverses only when it says ``bgr`` -- a LeRobot dataset stores
            RGB unless it says otherwise. The model always sees RGB.
    """
    import json

    root_dir = str(root_dir)
    with open(os.path.join(root_dir, 'meta', 'info.json')) as f:
        info = json.load(f)
    fps = info['fps']
    features = info.get('features', {})
    if reverse_channels is None:
        reverse_channels = info.get('channel_order', 'rgb').lower() != 'rgb'
    if camera_keys is None:
        found = {k for k, v in features.items() if v.get('dtype') in ('video', 'image')}
        # DROID's two views keep their trained order; anything else follows, deterministically.
        ordered = [k for k in DROID_CAMERA_ORDER if k in found]
        camera_keys = ordered + sorted(found - set(ordered))
    if not camera_keys:
        raise ValueError(f'no camera keys in {root_dir}/meta/info.json')
    # LeRobot stores cameras either as mp4 or as encoded frames in the parquet.
    cam_dtype = {cam: features.get(cam, {}).get('dtype', 'video') for cam in camera_keys}

    import pandas as pd

    episodes = pd.concat(
        [pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(root_dir, 'meta', 'episodes', '*', '*.parquet')))],
        ignore_index=True,
    ).sort_values('episode_index')
    frames = pd.concat(
        [pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(root_dir, 'data', '*', '*.parquet')))],
        ignore_index=True,
    ).sort_values('index')
    if task is not None:
        before = len(episodes)
        keep = episodes['tasks'].apply(
            lambda t: task in (list(t) if hasattr(t, '__len__') and not isinstance(t, str) else [t]))
        episodes = episodes[keep]
        if episodes.empty:
            raise ValueError(f'no episode with task {task!r} in {root_dir}')
        print(f'  task {task!r}: {len(episodes)}/{before} episodes')
    if max_episodes is not None:
        episodes = episodes.iloc[:max_episodes]

    has_reward = reward_key in frames.columns
    print(f'Loading {len(episodes)} LeRobot episodes from {root_dir} '
          f'({"recorded" if has_reward else "demonstration"} reward, cameras {camera_keys})...')

    # Pass 1: proprioception, actions, rewards, and where each episode's frames sit in which video.
    kept, segments = [], {}
    for _, ep in episodes.iterrows():
        lo, hi = int(ep['dataset_from_index']), int(ep['dataset_to_index'])
        rows = frames.iloc[lo:hi]
        traj_len = len(rows)
        if traj_len <= horizon_length:
            print(f"Warning: episode {int(ep['episode_index'])} has {traj_len} steps <= horizon. Skipping.")
            continue

        # Success means the reward rises above the step cost at the end.
        if has_reward:
            ep_reward = np.asarray(rows[reward_key].to_list(), dtype=np.float32).reshape(traj_len, -1)[:, 0]
            is_success = bool(ep_reward[-1] > ep_reward.min())
        else:
            is_success = True

        pos = len(kept)
        kept.append(
            dict(
                state=np.stack(rows[state_key].to_numpy()).astype(np.float32),
                actions=np.stack(rows[action_key].to_numpy()).astype(np.float32),
                traj_len=traj_len,
                is_success=is_success,
                lo=lo,
                hi=hi,
            )
        )
        for cam in camera_keys:
            if cam_dtype[cam] != 'video':
                continue
            key = (cam, int(ep[f'videos/{cam}/chunk_index']), int(ep[f'videos/{cam}/file_index']))
            start_frame = int(round(float(ep[f'videos/{cam}/from_timestamp']) * fps))
            segments.setdefault(key, []).append((pos, start_frame, traj_len))

    if not kept:
        raise ValueError(f'no usable episodes in {root_dir}')

    # Pass 2a: cameras stored as encoded frames come straight out of the data parquet.
    images = [[None] * len(camera_keys) for _ in kept]
    for i, cam in enumerate(camera_keys):
        if cam_dtype[cam] == 'video':
            continue
        for pos, ep in enumerate(kept):
            cells = frames.iloc[ep['lo']:ep['hi']][cam].to_numpy()
            images[pos][i] = _decode_frames(cells, image_size)

    # Pass 2: decode each video file once, in file order.
    cam_index = {cam: i for i, cam in enumerate(camera_keys)}
    for (cam, chunk_index, file_index), segs in tqdm(sorted(segments.items()), desc='Decoding video'):
        video_path = os.path.join(
            root_dir,
            info['video_path'].format(video_key=cam, chunk_index=chunk_index, file_index=file_index),
        )
        decoded = _decode_video_segments(
            video_path, [(start, length) for _, start, length in segs], image_size
        )
        for (pos, _, _), arr in zip(segs, decoded):
            images[pos][cam_index[cam]] = arr[..., ::-1] if reverse_channels else arr

    dataset = assemble_dataset(kept, images, camera_keys, sparse=sparse,
                               horizon_length=horizon_length,
                               compact_dataset=compact_dataset, discount=discount)
    return dataset

