import jax
import jax.numpy as jnp

def adaptive_l2_loss(error, p=0.5, c=1e-3, reduce='mean'):
    """Adaptive L2 loss.

    Args:
        error: (Batch, Total_Action_Dim) - Flattened error
        reduce: per-sample aggregation over the action dim. 'sum' matches the
            MeanFlow reference; 'mean' divides the error by the action dim, which
            raises the effective floor c by the same factor.
    """
    sq = jnp.square(error)
    squared_error = jnp.mean(sq, axis=-1) if reduce == 'mean' else jnp.sum(sq, axis=-1)

    w = 1.0 / (squared_error + c) ** p
    w = jax.lax.stop_gradient(w)

    loss = w * squared_error

    return loss.mean()

def sample_t_r(batch_size, rng, flow_ratio=0.5, time_dist='uniform'):
    # t >= r on the triangle; a (1 - flow_ratio) share gets r = t.
    rng, t_rng, r_rng = jax.random.split(rng, 3)

    if time_dist == 'beta':
        # Beta(1.5, 1) puts more mass near the decode point t = 1.
        t = jax.random.beta(t_rng, 1.5, 1.0, [batch_size, 1])
        r = jax.random.beta(r_rng, 1.5, 1.0, [batch_size, 1])
    else:
        t = jax.random.uniform(t_rng, [batch_size, 1])
        r = jax.random.uniform(r_rng, [batch_size, 1])
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
