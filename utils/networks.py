from typing import Any, Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp
import flax

def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0, 'intermediates': 0},
        split_rngs={'params': True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class FourierFeatures(nn.Module):
    # used for timestep embedding
    output_size: int = 64
    learnable: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.learnable:
            w = self.param('kernel', nn.initializers.normal(0.2),
                           (self.output_size // 2, x.shape[-1]), jnp.float32)
            f = 2 * jnp.pi * x @ w.T
        else:
            half_dim = self.output_size // 2
            f = jnp.log(10000) / (half_dim - 1)
            f = jnp.exp(jnp.arange(half_dim) * -f)
            f = x * f
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)


def zero_init():
    return nn.initializers.zeros

class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
            if i == len(self.hidden_dims) - 2:
                self.sow('intermediates', 'feature', x)
        return x

class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return jnp.exp(log_value)


class Value(nn.Module):
    """Value/critic network.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 1
    encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        value_net = mlp_class((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)

        self.value_net = value_net

    def __call__(self, observations, actions=None):
        """Return values or critic values.

        Args:
            observations: Observations.
            actions: Actions (optional).
        """
        if self.encoder is not None:
            if isinstance(observations, (dict, flax.core.FrozenDict)):
                img_embed = self.encoder(observations['image'])
                state = observations['state']
                inputs = [img_embed, state]
            else:
                inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs).squeeze(-1)

        return v

class ActorVectorField(nn.Module):
    """Actor vector field network for flow matching.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None
    use_fourier_features: bool = False
    fourier_feature_dim: int = 64
    latent_dist: str = 'normal'
    # sphere_exp only.
    log_std_min: float = -5.0
    log_std_max: float = 0.0
    log_std_init: Optional[float] = None

    def setup(self) -> None:
        # 'sphere_exp' emits one extra scalar for log_std.
        _extra = 1 if self.latent_dist == 'sphere_exp' else 0
        _out = self.action_dim + _extra
        self.mlp = MLP((*self.hidden_dims, _out), activate_final=False, layer_norm=self.layer_norm)
        if self.use_fourier_features:
            self.ff = FourierFeatures(self.fourier_feature_dim)

    @nn.compact
    def __call__(self, observations, actions=None, times=None, v_base=None, g=None, is_encoded=False, rng=None, return_logp=False):
        """Return the vectors at the given states, actions, and times (optional).

        Args:
            observations: Observations.
            actions: Actions.
            times: Times (optional).
            v_base: Base noise; for 'sphere_exp' it is the tangent sample, otherwise an extra input.
            g: Extra conditioning input (optional).
            is_encoded: Whether the observations are already encoded.
            return_logp: For 'sphere_exp', also return the log-density and log-std.
        """

        inputs = []
        if self.encoder is not None:
            if isinstance(observations, (dict, flax.core.FrozenDict)):
                if not is_encoded:
                    img_embed = self.encoder(observations['image'])
                else:
                    img_embed = observations['image']
                state = observations['state']
                inputs = [img_embed, state]
            else:
                if not is_encoded:
                    inputs = [self.encoder(observations)]
                else:
                    inputs = [observations]
        else:
            inputs = [observations]

        if actions is not None:
            inputs.append(actions)

        if times is not None:
            if self.use_fourier_features:
                times = self.ff(times)
            inputs.append(times)

        if v_base is not None and self.latent_dist != 'sphere_exp':
            inputs.append(v_base)

        if g is not None:
            inputs.append(g)

        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.mlp(inputs)

        if self.latent_dist == 'normal':
            return v
        elif self.latent_dist == 'truncated_normal':
            return 2 * nn.tanh(v)
        elif self.latent_dist == 'uniform':
            return nn.tanh(v)
        elif self.latent_dist == 'simplex':
            return nn.softmax(v)
        elif self.latent_dist == 'sphere':
            sq_sum = jnp.sum(jnp.square(v), axis=-1, keepdims=True)
            norm = jnp.sqrt(sq_sum + 1e-6) 
            return v / norm * jnp.sqrt(self.action_dim)
        elif self.latent_dist == 'sphere_exp':
            # Tangent Gaussian pushed through the exponential map.
            #   t = std * (eps - (eps.muh) muh);  th = ||t||/sqrt(d)
            #   z = cos(th) mu + sin(th) sqrt(d) t/||t||
            d = self.action_dim
            mu = v[..., :d]
            mu = mu / (jnp.linalg.norm(mu, axis=-1, keepdims=True) + 1e-8) * jnp.sqrt(d)
            muh = mu / jnp.sqrt(d)
            _lo, _hi = self.log_std_min, self.log_std_max
            _raw = v[..., d:d + 1]
            if self.log_std_init is not None:
                _u = 2.0 * (self.log_std_init - _lo) / (_hi - _lo) - 1.0
                _raw = _raw + jnp.arctanh(jnp.clip(_u, -0.999, 0.999))
            log_std = _lo + 0.5 * (_hi - _lo) * (nn.tanh(_raw) + 1.0)     # (..., 1)
            if v_base is None:
                # Deterministic action: the mode.
                return (mu, jnp.zeros(mu.shape[:-1]), log_std[..., 0]) if return_logp else mu
            std = jnp.exp(log_std)
            e = std * v_base
            t = e - jnp.sum(e * muh, axis=-1, keepdims=True) * muh
            tn = jnp.linalg.norm(t, axis=-1, keepdims=True) + 1e-12
            th = jnp.clip(tn / jnp.sqrt(d), 1e-6, jnp.pi - 1e-3)
            z = jnp.cos(th) * mu + jnp.sin(th) * jnp.sqrt(d) * (t / tn)
            _s = log_std[..., 0]
            log_prob = (-0.5 * jnp.sum(t * t, axis=-1) / jnp.exp(2.0 * _s)
                        - (d - 1.0) * _s
                        - 0.5 * (d - 1.0) * jnp.log(2.0 * jnp.pi)
                        - (d - 2.0) * jnp.log(jnp.sin(th[..., 0]) / th[..., 0]))
            return (z, log_prob, _s) if return_logp else z
        elif self.latent_dist == 'beta':
            return 2 * nn.tanh(v)
        return v
    
class ActorMeanFlowField(nn.Module):
    """MeanFlow field network: an MLP conditioned on two times, t and r.

    With ``use_fourier_features`` both times are embedded with Fourier features before they are
    concatenated to the input.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None
    use_fourier_features: bool = False
    fourier_feature_dim: int = 16

    def setup(self) -> None:
        # Standard MLP for the vector field
        self.mlp = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)
        # Fourier Features for Timestep (t)
        if self.use_fourier_features:
            self.ff = FourierFeatures(self.fourier_feature_dim)

    @nn.compact
    def __call__(self, observations, actions, t, r=None, g=None, is_encoded=False):
        """Return the field at the given observations, actions and times.

        Args:
            observations: Observations (Batch, Obs_Dim).
            actions: Actions (Batch, Act_Dim).
            t: First time input (optional).
            r: Second time input (optional).
            g: Extra conditioning input (optional).
            is_encoded: Whether the observations are already encoded.
        """
        # 1. Encode observations if necessary
        inputs_list = []
        if self.encoder is not None:
            if isinstance(observations, (dict, flax.core.FrozenDict)):
                if not is_encoded:
                    img_embed = self.encoder(observations['image'])
                else:
                    img_embed = observations['image']
                state = observations['state']
                inputs_list = [img_embed, state]
            else:
                inputs_list = [self.encoder(observations)]
        else:
            inputs_list = [observations]

        # 2. Construct input list
        inputs_list.append(actions)

        if t is not None:
            if self.use_fourier_features:
                t = self.ff(t)
            inputs_list.append(t)

        if r is not None:
            if self.use_fourier_features:
                r = self.ff(r)
            inputs_list.append(r)

        if g is not None:
            inputs_list.append(g)

        # 3. Concatenate all inputs along the last dimension
        inputs = jnp.concatenate(inputs_list, axis=-1)

        # 4. Pass through the MLP
        v = self.mlp(inputs)

        return v