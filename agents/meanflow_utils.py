import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

def adaptive_l2_loss(error, p=0.5, c=1e-3):
    """
    Adaptive L2 loss with Valid Masking.
    Args:
        error: (Batch, Total_Action_Dim) - Flattened error
    """
    ## THIS SHOULD BE modified so support horizion
    # 1. Calculate Error per Sample (Sum of Squared Error)
    # Aggregated over Action Dim axis
    squared_error = jnp.mean(jnp.square(error), axis=-1)
    
    # 2. Calculate Adaptive Weight (Use stop_gradient so gradient doesn't flow)
    # p = 1 - gamma (Similar to norm_p=1.0 in official code)
    w = 1.0 / (squared_error + c) ** p
    w = jax.lax.stop_gradient(w)
    
    # 3. Calculate Weighted Loss
    # loss = w * ||u - u_tgt||^2
    loss = w * squared_error
    
    return loss.mean()

def sample_t_r(batch_size, rng, flow_ratio=0.25):
    # lognorm sampling (Seems working better than uniform)
    rng, t_rng, r_rng = jax.random.split(rng, 3)

    t = jax.nn.sigmoid(jax.random.normal(t_rng, [batch_size, 1]) - 0.4)
    r = jax.nn.sigmoid(jax.random.normal(r_rng, [batch_size, 1]) - 0.4)
    t, r = jnp.maximum(t, r), jnp.minimum(t, r)

    data_size = int(batch_size * (1 - flow_ratio))
    zero_mask = jnp.arange(batch_size) < data_size
    zero_mask = zero_mask.reshape(batch_size, 1)
    r = jnp.where(zero_mask, t, r)

    return t, r

def sample_latent_dist(x_rng, sample_shape, latent_dist='sphere'):
    if latent_dist == 'normal':
        e = jax.random.normal(x_rng, sample_shape)
    elif latent_dist == 'truncated_normal':
        e = jax.random.truncated_normal(x_rng, -2.0, 2.0, shape=sample_shape)
    elif latent_dist == 'uniform':
        e = jax.random.uniform(x_rng, sample_shape, minval=-1.0, maxval=1.0)
    elif latent_dist == 'sphere':
        e = jax.random.normal(x_rng, sample_shape)
        sq_sum = jnp.sum(jnp.square(e), axis=-1, keepdims=True)
        norm = jnp.sqrt(sq_sum + 1e-6)
        e = e / norm * jnp.sqrt(sample_shape[-1])
    elif latent_dist == 'sphere_plus':
        action_dim = sample_shape[-1]
        sample_shape = sample_shape[:-1] + (action_dim + 1,)
        e = jax.random.normal(x_rng, sample_shape)
        e_norm = jnp.linalg.norm(e, axis=-1, keepdims=True)
        e = e / (e_norm + 1e-6)
        e = e[..., :-1]
    return e
