import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces
import io
from utils.datasets import Dataset
import wandb
from sklearn.decomposition import PCA  # [New] PCA for visualization

def _get_bandit6_centers(rng=None):
    """
    Generate 9 fixed cluster centers in a 50-dimensional space.
    The first center (Index 0) is set as the Optimal Target.
    """
    # Use a fixed seed to ensure centers are always at the same positions regardless of the outer seed
    local_rng = np.random.RandomState(42) 
    
    # 50-dimensional, 9 clusters
    # Randomly placed within [-0.8, 0.8]
    centers = local_rng.uniform(-0.8, 0.8, size=(9, 50))
    return centers

def generate_bandit_1(n, rng):
    # Lv 1: 4-Gaussians
    centers = [(-0.7, -0.7), (-0.7, 0.7), (0.7, -0.7), (0.7, 0.7)]
    data = []
    for _ in range(n):
        c = centers[rng.randint(4)]
        data.append(np.array(c) + 0.1 * rng.randn(2))
    return np.array(data).astype(np.float32)

def generate_bandit_2(n, rng):
    # Lv 2: Checkerboard
    data = []
    while len(data) < n:
        x, y = rng.uniform(-1, 1, 2)
        if (np.floor(x * 2) % 2 + np.floor(y * 2) % 2) % 2 == 0:
             data.append([x, y])
    return np.array(data[:n]).astype(np.float32)

def generate_bandit_3(n, rng):
    # Lv 3: Two Moons (Hard) - Corrected
    # Exact Interleaving Moons implementation
    n_upper = n // 2
    n_lower = n - n_upper
    
    # 1. Upper Moon (t: 0 ~ pi) -> center (0, 0)
    t_upper = np.linspace(0, np.pi, n_upper)
    x_upper = np.cos(t_upper)
    y_upper = np.sin(t_upper)
    upper_moon = np.stack([x_upper, y_upper], axis=1)
    
    # 2. Lower Moon (t: 0 ~ pi) -> center (1, -0.5)
    t_lower = np.linspace(0, np.pi, n_lower)
    x_lower = 1 - np.cos(t_lower)
    y_lower = 1 - np.sin(t_lower) - 0.5
    lower_moon = np.stack([x_lower, y_lower], axis=1)
    
    # 3. Combine
    data = np.vstack([upper_moon, lower_moon])
    
    # 4. Normalize to [-1, 1]
    # Original range: x[-1, 2], y[-0.5, 1] -> center (0.5, 0.25)
    # Shift center to origin and scale
    data[:, 0] -= 0.5
    data[:, 1] -= 0.25
    data *= 0.6 # Shrink to fit safely within [-1, 1]
    
    # 5. Add Noise
    data += 0.05 * rng.randn(*data.shape)
    
    return np.clip(data, -1.0, 1.0).astype(np.float32)

def generate_bandit_4(n, rng):
    # Lv 4: Rings
    noisy_circles = []
    for _ in range(n):
        radius = 0.8 if rng.rand() > 0.5 else 0.3
        angle = rng.rand() * 2 * np.pi
        noise = 0.05 * rng.randn(2)
        point = np.array([np.cos(angle) * radius, np.sin(angle) * radius]) + noise
        noisy_circles.append(point)
    return np.array(noisy_circles).astype(np.float32)

def generate_bandit_5(n, rng):
    # Lv 5: Spiral
    t = np.sqrt(rng.uniform(0, 1, n)) * 540 * (2 * np.pi) / 360
    x = -np.cos(t) * t + rng.rand(n) * 0.05
    y = np.sin(t) * t + rng.rand(n) * 0.05
    data = np.vstack([x, y]).T
    data = data / (np.max(np.abs(data)) + 1e-6)
    return data.astype(np.float32)

def generate_bandit_6(n, rng):
    """
    Lv 6: High-Dim Clusters (Action Chunking Sim)
    - Dim: 50
    - Clusters: 9 (Mixture of Gaussians)
    - Reward: 1.0 only near Center 0, 0.0 otherwise
    """
    centers = _get_bandit6_centers()
    n_clusters = len(centers)
    
    data = []
    for _ in range(n):
        # Select one of 9 clusters (Uniform)
        # i.e., the dataset contains correct (1/9) and incorrect (8/9) samples
        c_idx = rng.randint(n_clusters)
        center = centers[c_idx]
        
        # Noise (Cluster Spread)
        # In 50 dimensions, large noise can cause cluster overlap. Keep it small.
        noise = 0.05 * rng.randn(50)
        data.append(center + noise)
        
    return np.array(data).astype(np.float32)

# Name mapping
GENERATORS = {
    'bandit-1': generate_bandit_1,
    'bandit-2': generate_bandit_2,
    'bandit-3': generate_bandit_3,
    'bandit-4': generate_bandit_4,
    'bandit-5': generate_bandit_5,
    'bandit-6': generate_bandit_6,
}

# ==========================================
# 2. Reward Functions
# ==========================================

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces
import io

# Reward function (vectorized)
def get_reward_batch(env_name, actions):
    """
    Multi-modal Reward with Hard Peaks.
    Ensures Optimal Regions get exactly 1.0 reward.
    """
    actions = np.atleast_2d(actions)
    rewards = np.zeros(len(actions), dtype=np.float32)
    
    x = actions[:, 0]
    y = actions[:, 1]

    if env_name == 'bandit-1': # 4-Gaussians
        # Centers
        c1 = np.array([0.7, 0.7])   # Optimal
        c2 = np.array([0.7, -0.7])  # Trap 1
        c3 = np.array([-0.7, 0.7])  # Trap 2
        c4 = np.array([-0.7, -0.7]) # Trap 3
        
        # Distances
        d1 = np.linalg.norm(actions - c1, axis=1)
        d2 = np.linalg.norm(actions - c2, axis=1)
        d3 = np.linalg.norm(actions - c3, axis=1)
        d4 = np.linalg.norm(actions - c4, axis=1)
        
        # Thresholding (within radius 0.3)
        # Overlap is unlikely, but apply priority ordering
        rewards[d4 < 0.3] = 0.2
        rewards[d3 < 0.3] = 0.4
        rewards[d2 < 0.3] = 0.7
        rewards[d1 < 0.3] = 1.0 # Optimal always gets 1.0

    elif env_name == 'bandit-2': # Checkerboard
        # Check Pattern
        in_checker = (np.floor(x * 2) % 2 + np.floor(y * 2) % 2) % 2 == 0
        
        # 1. Optimal Zone: upper-right box (0.5~1.0, 0.5~1.0)
        is_optimal = (x > 0.5) & (x < 1.0) & (y > 0.5) & (y < 1.0)
        
        # 2. Sub-optimal: remaining checkerboard cells
        is_subopt = in_checker & (~is_optimal)
        
        rewards[is_subopt] = 0.5
        rewards[is_optimal] = 1.0

    elif env_name == 'bandit-3': # Two Moons
        # 1. Upper Moon Tip (Optimal)
        t_tip = np.linspace(0, np.pi * 0.2, 50)
        tip_x = (np.cos(t_tip) - 0.5) * 0.6
        tip_y = (np.sin(t_tip) - 0.25) * 0.6
        tip_arc = np.stack([tip_x, tip_y], axis=1)
        dist_tip = np.min(np.linalg.norm(actions[:, None, :] - tip_arc[None, :, :], axis=2), axis=1)
        
        # 2. Rest of Upper Moon (Sub-optimal)
        t_up = np.linspace(np.pi * 0.2, np.pi, 100)
        up_x = (np.cos(t_up) - 0.5) * 0.6
        up_y = (np.sin(t_up) - 0.25) * 0.6
        up_arc = np.stack([up_x, up_y], axis=1)
        dist_up = np.min(np.linalg.norm(actions[:, None, :] - up_arc[None, :, :], axis=2), axis=1)

        # 3. Lower Moon (Bad Trap)
        t_low = np.linspace(0, np.pi, 100)
        low_x = (1 - np.cos(t_low) - 0.5) * 0.6
        low_y = (1 - np.sin(t_low) - 0.5 - 0.25) * 0.6
        low_arc = np.stack([low_x, low_y], axis=1)
        dist_low = np.min(np.linalg.norm(actions[:, None, :] - low_arc[None, :, :], axis=2), axis=1)
        
        # Assign Rewards (Threshold 0.1)
        rewards[dist_low < 0.1] = 0.2
        rewards[dist_up < 0.1] = 0.6
        rewards[dist_tip < 0.1] = 1.0 # Tip always gets 1.0

    elif env_name == 'bandit-4': # Rings
        r = np.linalg.norm(actions, axis=1)
        theta = np.arctan2(actions[:, 1], actions[:, 0])
        
        # Conditions
        is_inner = (r > 0.2) & (r < 0.4) # Inner Ring (r~0.3)
        is_outer = (r > 0.7) & (r < 0.9) # Outer Ring (r~0.8)
        is_quad1 = (theta > np.pi*2/6) & (theta < np.pi*3/6) # First quadrant
        
        rewards[is_inner] = 0.3
        rewards[is_outer] = 0.6 # All outer ring gets 0.6
        rewards[is_outer & is_quad1] = 1.0 # Only first quadrant outer ring gets 1.0

    elif env_name == 'bandit-5': # Spiral
        # 1. Optimal Tip (t: 2.5 ~ 3.0)
        t_tip = np.linspace(2.5, 3.0, 100)
        tx = t_tip * np.cos(3 * t_tip) * 0.3
        ty = t_tip * np.sin(3 * t_tip) * 0.3
        tip_arc = np.stack([tx, ty], axis=1)
        dist_tip = np.min(np.linalg.norm(actions[:, None, :] - tip_arc[None, :, :], axis=2), axis=1)
        
        # 2. Body (t: 0.5 ~ 2.5)
        t_body = np.linspace(0.5, 2.5, 200)
        bx = t_body * np.cos(3 * t_body) * 0.3
        by = t_body * np.sin(3 * t_body) * 0.3
        body_arc = np.stack([bx, by], axis=1)
        dist_body = np.min(np.linalg.norm(actions[:, None, :] - body_arc[None, :, :], axis=2), axis=1)
        
        # Assign reward if on the spiral trajectory (continuous gradient style)
        # But 1.0 is only assigned to the tip
        rewards[dist_body < 0.1] = 0.5 # Body gets 0.5
        rewards[dist_tip < 0.1] = 1.0  # Tip gets 1.0

    elif env_name == 'bandit-6': # High-Dim Clusters
        centers = _get_bandit6_centers()
        dists = np.linalg.norm(actions[:, None, :] - centers[None, :, :], axis=2) # (N, 9)
        min_dists = np.min(dists, axis=1)
        closest_idx = np.argmin(dists, axis=1)
        
        # Only assign reward if within threshold 0.3
        valid_mask = min_dists < 0.3
        
        # Default score (Noise)
        scores = np.full(len(actions), 0.0, dtype=np.float32)
        
        # Per-cluster score mapping
        # Cluster 0: 1.0 (Optimal)
        # Cluster 1: 0.8
        # Rest: 0.1
        cluster_scores = np.full(9, 0.1, dtype=np.float32)
        cluster_scores[0] = 1.0
        cluster_scores[1] = 0.8
        
        # Assignment
        # Only assign scores to valid samples
        assigned_scores = cluster_scores[closest_idx]
        rewards = np.where(valid_mask, assigned_scores, 0.0)

    return rewards


class ToyBanditEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, env_name, seed=0, render_mode=None):
        super().__init__()
        self.env_name = env_name
        if env_name == 'bandit-6':
            self.action_dim = 50
            self.obs_dim = 2
        else:
            self.action_dim = 2
            self.obs_dim = 2
        
        self.observation_space = spaces.Box(-1, 1, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, shape=(self.action_dim,), dtype=np.float32)
        
        self.render_mode = render_mode
        self.fig, self.ax = None, None
        
        # Store as list or array for batch visualization
        self.last_actions = None
        self.last_rewards = None
        
        rng = np.random.RandomState(seed)
        self.gt_data = GENERATORS[env_name](2000, rng)

        # 2. [New] PCA Initialization (Fixed Mapping)
        self.pca = None
        if self.action_dim > 2:
            # Fit PCA axes on GT data and fix them.
            # This ensures agent-generated data is also projected onto the same fixed axes.
            self.pca = PCA(n_components=2)
            self.pca.fit(self.gt_data)
            
            # Also transform GT data to 2D (for visualization)
            self.gt_data_2d = self.pca.transform(self.gt_data)
        else:
            self.gt_data_2d = self.gt_data

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.obs_dim, dtype=np.float32), {}

    def step(self, action):
        # 1. Action Clipping & Batch Handling
        # Input can be either (2,) or (Batch, 2)
        action = np.array(action)
        is_batch = action.ndim > 1
        
        # Clip
        action = np.clip(action, -1.0, 1.0)
        
        # 2. Reward Calculation (Vectorized)
        # Use vectorized get_reward_batch for batch processing
        reward = get_reward_batch(self.env_name, action)
        
        # 3. Save for Rendering
        # Save for later rendering
        self.last_actions = action
        self.last_rewards = reward
        
        # 4. Return Construction
        # One-step termination
        # Return matching the batch size
        if is_batch:
            batch_size = len(action)
            obs = np.zeros((batch_size, self.obs_dim), dtype=np.float32)
            terminated = np.ones(batch_size, dtype=bool)
            truncated = np.zeros(batch_size, dtype=bool)
            return obs, reward.astype(float), terminated, truncated, {}
        else:
            return np.zeros(self.obs_dim, dtype=np.float32), float(reward), True, False, {}

    def render(self):
        # 1. Style and canvas setup
        plt.style.use('default')
        
        # Create 2 subplots (1x2 Layout)
        if self.fig is None:
            # Total size (12, 6) -> each plot (6, 6) keeping square aspect
            self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
            
            # Background color: ivory
            bg_color = '#FFFDF5'
            self.fig.patch.set_facecolor(bg_color)
            for ax in self.axes:
                ax.set_facecolor(bg_color)
                
        # Clear axes for each render call
        for ax in self.axes:
            ax.clear()
            ax.axis('off')
            
        # Zoom settings
        limit = 1.2 if not self.pca else 2.5
        for ax in self.axes:
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)

        # Title settings
        self.axes[0].set_title("Dataset Density (Offline Data)", fontsize=14, fontweight='bold', color='#8B8000')
        self.axes[1].set_title("Current Policy (Agent)", fontsize=14, fontweight='bold', color='#00008B')

        # Import required utilities
        from scipy.stats import gaussian_kde

        # Create common grid
        grid_res = 100
        x_grid = np.linspace(-limit, limit, grid_res)
        y_grid = np.linspace(-limit, limit, grid_res)
        gx, gy = np.meshgrid(x_grid, y_grid)
        positions = np.vstack([gx.ravel(), gy.ravel()]) 

        # ==========================================
        # Plot 1: Dataset Density (Left) - Yellow/Orange
        # ==========================================
        ax_data = self.axes[0]
        
        if self.gt_data_2d is not None:
            try:
                # Jittering & Bandwidth Fix
                noise = np.random.normal(0, 0.02, size=self.gt_data_2d.shape)
                data_safe = self.gt_data_2d + noise
                
                data_kde = gaussian_kde(data_safe.T)
                data_kde.set_bandwidth(bw_method=0.2)
                
                z_data = data_kde(positions).reshape(grid_res, grid_res)
                z_data_norm = z_data / (z_data.max() + 1e-8)
                
                # YlOrBr: Yellow -> Orange
                ax_data.contourf(gx, gy, z_data_norm, levels=np.linspace(0.1, 1.0, 12), cmap='YlOrBr', alpha=0.8)
                
            except Exception:
                ax_data.scatter(self.gt_data_2d[:, 0], self.gt_data_2d[:, 1], c='orange', s=5, alpha=0.3)

        # ==========================================
        # Plot 2: Current Action Density (Right) - Blue
        # ==========================================
        ax_policy = self.axes[1]
        
        # # Ground Truth Context (light gray background for reference)
        # if self.action_dim == 2:
        #      points_bg = np.vstack([gx.ravel(), gy.ravel()]).T
        #      rewards_bg = get_reward_batch(self.env_name, points_bg).reshape(grid_res, grid_res)
        #      # Lightly show the optimal region
        #      ax_policy.contourf(gx, gy, rewards_bg, levels=[0.5, 2.0], colors=['#E0E0E0'], alpha=0.3)

        # Action Density (KDE)
        if self.last_actions is not None:
            actions = np.atleast_2d(self.last_actions)
            if self.pca:
                actions_2d = self.pca.transform(actions)
            else:
                actions_2d = actions
            
            try:
                # Jittering & Bandwidth Fix
                noise = np.random.normal(0, 0.02, size=actions_2d.shape)
                actions_safe = actions_2d + noise
                
                kde = gaussian_kde(actions_safe.T)
                kde.set_bandwidth(bw_method=0.25)
                
                z_act = kde(positions).reshape(grid_res, grid_res)
                z_act_norm = z_act / (z_act.max() + 1e-8)
                
                # Blues: Blue Density
                ax_policy.contourf(gx, gy, z_act_norm, levels=np.linspace(0.1, 1.0, 10), cmap='Blues', alpha=0.8)
                
            except Exception:
                ax_policy.scatter(actions_2d[:, 0], actions_2d[:, 1], c='blue', s=10, alpha=0.5)
            
        # ==========================================
        # Capture logic (buffer_rgba)
        # ==========================================
        self.fig.canvas.draw()
        img_rgba = np.asarray(self.fig.canvas.buffer_rgba())
        img_arr = np.array(img_rgba[:, :, :3], dtype=np.uint8)
            
        return img_arr
            
    def close(self):
        if self.fig: plt.close(self.fig)

def make_bandit_datasets(env_name, dataset_size=100000, seed=0):
    """
    Toy Bandit dataset generation function.
    Uses get_reward_batch for fast vectorized reward computation.
    """
    # Create environment (for metadata and space verification)  
    
    # 1. Generate Actions (using Manifold Generator)
    # Shape: (dataset_size, 2)
    # Create local RandomState object (Local RNG)
    rng = np.random.RandomState(seed)
    
    # 1. Generate Actions (using Manifold Generator)
    actions = GENERATORS[env_name](dataset_size, rng)
    
    # 2. Compute Rewards (Vectorized)
    # Removed list comprehension -> use batch function
    rewards = get_reward_batch(env_name, actions)

    # 3. Dummy fields (match Offline RL format)
    # Observations are meaningless, so fill with zeros
    observations = np.zeros((dataset_size, 2), dtype=np.float32)
    next_observations = np.zeros_like(observations)
    
    # Bandit: every step terminates (Terminal=1, Mask=0)
    terminals = np.zeros(dataset_size, dtype=np.float32)
    masks = np.zeros(dataset_size, dtype=np.float32)
    
    data_dict = {
        'observations': observations,
        'actions': actions,
        'rewards': rewards,
        'next_observations': next_observations,
        'terminals': terminals,
        'masks': masks
    }
    
    # Return Dataset object
    dataset = Dataset.create(**data_dict)

    sr = (rewards == 1.0).sum() / (rewards > -1).sum()
    
    print(f"Dataset ({env_name}) Created: {dataset_size} samples")
    print(f"SR - : {sr*100:.2f}%")
    
    # Return dataset matching the structure expected by main.py
    return dataset


import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, rng=key, **kwargs)

    return wrapped

def evaluate(
    agent,
    env,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    eval_gaussian=None,
    action_shape=None,
    observation_shape=None,
    action_dim=None,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))

    observation, info = env.reset()
    observations = np.zeros((1000, 2), dtype=np.float32)
    actions = actor_fn(observations=observations)
    actions = np.array(actions).reshape(1000, -1)
    _, rewards, _, _, _ = env.step(actions)

    # 4. Calculate Metrics
    success_rate = np.sum(rewards==1.0) / np.ones_like(rewards).sum() # Reward is 0 or 1
    avg_reward = np.mean(rewards)
    
    stats = {
        "eval/success_rate": success_rate,
        "eval/avg_reward": avg_reward,
    }
    img_array = env.render()

    if img_array is not None:
        # WandB Image logging (channel order: HWC -> CHW not needed, wandb.Image handles it)
        # PyTorch/Tensorboard style CHW conversion is also common.
        # Here we pass HWC directly.
        stats["eval/policy_distribution"] = wandb.Image(
            img_array, 
            caption=f"Success Rate {success_rate*100:.1f}%"
        )

    return stats, _, _

