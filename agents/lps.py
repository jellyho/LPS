import copy
import functools
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import ml_collections

from agents.meanflow_utils import adaptive_l2_loss, sample_t_r, sample_latent_dist
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field, get_batch_shape, make_optimizer
from utils.networks import ActorMeanFlowField, Value, ActorVectorField, LogParam
from utils.dit import mf_dit_models

class LPSAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the critic TD loss."""
        # Q(s, a) <- R + \gamma^H Q(s', a'), with a' drawn from the steered policy
        batch_size = batch['actions'].shape[0]

        rng, sample_rng = jax.random.split(rng, 2)
        next_actions = self.sample_actions(batch['next_observations'], rng=sample_rng)

        next_qs = self.network.select('target_critic')(
            batch['next_observations'], 
            actions=next_actions.reshape(batch_size, -1)
        )

        if self.config['num_critic'] > 1:
            # Pessimistic TD target: ensemble mean minus _CRITIC_RHO standard deviations.
            next_q = jnp.mean(next_qs, axis=0) - _CRITIC_RHO * jnp.std(next_qs, axis=0)
        else:
            next_q = next_qs
        
        qs = self.network.select('critic')(
            batch['observations'], 
            actions=batch['actions'].reshape(batch_size, -1),
            params=grad_params
        )

        target_q = batch['rewards'] + \
            (self.config['discount'] ** self.config["horizon_length"]) * \
        batch['masks'] * next_q

        critic_loss = jnp.square(
            qs - jnp.broadcast_to(target_q, qs.shape)
        ).mean()

        q = qs.mean(axis=0) # For logging

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the MeanFlow BC loss for the base policy."""
        batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))  # fold in horizon_length together with action_dim
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng, drop_rng = jax.random.split(rng, 4)

        # BC mean flow loss.
        x = batch_actions

        t, r = sample_t_r(batch_size, t_rng, self.config['flow_ratio'],
                          self.config['time_dist'])

        # The latent need not be Gaussian; latent_dist selects it
        e = sample_latent_dist(x_rng, (batch_size, action_dim), self.config['latent_dist'])
        z = (1 - t) * x + t * e
        v = e - x

        def mean_flow_forward(z, t, r):
            return self.network.select('actor_bc_flow')(
                batch['observations'], 
                z, 
                t,
                t - r,
                params=grad_params
            )

        x_pred, dxdt = jax.jvp(
            mean_flow_forward, 
            (z, t, r), 
            (v, jnp.ones_like(t), jnp.zeros_like(r))
        )

        u, dudt = z - x_pred, v - dxdt
        # The JVP is unbounded and multiplies the gap, so it can swamp the target.
        dudt = jnp.clip(dudt, -1.0, 1.0)

        u_tgt = v - jnp.clip(t - r, a_min=0.0, a_max=1.0) * dudt
        u_tgt = jax.lax.stop_gradient(u_tgt)
        err = u - u_tgt
        loss = adaptive_l2_loss(err, reduce=self.config['adaptive_reduce'])

        return loss, {
            'actor_loss': loss,
        }

    def latent_actor_loss(self, batch, grad_params, rng):
        observations = batch['observations']
        batch_size = batch['actions'].shape[0]
        latent_dim = self.config['action_dim'] * self.config['horizon_length']
        
        ### Query latent actor
        rng, l_rng, e_rng = jax.random.split(rng, 3)
        _eps = jax.random.normal(e_rng, (batch_size, latent_dim))
        z_pred, sac_logp, sac_log_std = self.network.select('latent_actor')(
            observations,
            v_base=_eps,
            rng=l_rng,
            return_logp=True,
            params=grad_params
        )

        x_pred = self.compute_flow_actions(observations, z_pred)
        a_pred_flat = jnp.reshape(x_pred, (batch_size, latent_dim))

        info_dict = {
            'z_norm': jnp.mean(jnp.square(z_pred)),
        }

        qs = self.network.select('critic')(observations, a_pred_flat.reshape(batch_size, -1))
        if self.config['num_critic'] > 1:
            q = jnp.mean(qs, axis=0)  # The latent actor climbs the ensemble mean.
        else:
            q = qs
        lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
        alpha_sac = self.network.select('latent_alpha')(params=grad_params)
        # Per tangent dimension, the entropy term matches the |Q|-normalised Q term in scale.
        ent_scale = 1.0 / (latent_dim - 1) if self.config.get('latent_entropy_per_dim', False) else 1.0
        loss = -q.mean() + jax.lax.stop_gradient(alpha_sac) * ent_scale * sac_logp.mean()
        loss = self.config['alpha'] * lam * loss
        ent = -jax.lax.stop_gradient(sac_logp).mean()
        tgt = _target_entropy(latent_dim, self.config['latent_target_angle_deg'])
        loss = loss + (alpha_sac * (ent - tgt)).mean()
        info_dict['sac_entropy'] = ent
        info_dict['sac_entropy_target'] = jnp.asarray(tgt, dtype=ent.dtype)
        info_dict['sac_alpha'] = alpha_sac
        info_dict['sac_log_std'] = sac_log_std.mean()
        _lo, _hi = _LOG_STD_MIN, self.config['latent_log_std_max']
        _m = 0.05 * (_hi - _lo)
        info_dict['sac_frac_at_clip'] = (
            (sac_log_std <= _lo + _m) | (sac_log_std >= _hi - _m)).mean()

        info_dict['latent_loss'] = loss

        return loss, info_dict
    
    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        loss = 0

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        loss += critic_loss
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        loss += actor_loss
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        latent_loss, latent_info = self.latent_actor_loss(batch, grad_params, actor_rng)
        loss += latent_loss
        for k, v in latent_info.items():
            info[f'latent/{k}'] = v

        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @staticmethod
    def _update(agent, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, 'critic')
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)
    
    @jax.jit
    def sample_values(
        self,
        observations,
        rng=None,
    ):
        return jnp.zeros_like(observations)
    
    @functools.partial(jax.jit, static_argnames=('steps',))
    def sample_actions(
        self,
        observations,
        rng=None,
        steps=1,
    ):
        """Sample action chunks: the latent actor picks a latent, the BC flow decodes it."""
        latent_dim = self.config["horizon_length"] * self.config["action_dim"]
        batch_shape = get_batch_shape(observations, self.config['leaf_ndims'])

        rng, s_rng = jax.random.split(rng, 2)
        _e = jax.random.normal(s_rng, (*batch_shape, latent_dim))
        noises = self.network.select('latent_actor')(observations, v_base=_e)
        actions = self.compute_flow_actions(observations, noises, steps=steps)
        actions = jnp.reshape(
            actions, 
            (*batch_shape, self.config["horizon_length"], self.config["action_dim"])
        )
        return actions

    @functools.partial(jax.jit, static_argnames=('steps',))
    def compute_flow_actions(
        self,
        observation,
        noise,
        steps=1,
    ):
        """Decode the latent into an action with `steps` Euler steps."""
        z = noise
        for i in range(steps):
            t = jnp.full_like(noise[..., :1], 1.0 - i / steps)
            dt = jnp.full_like(t, 1.0 / steps)
            x_pred = self.network.select('actor_bc_flow')(observation, z, t, dt)
            z = (1.0 - dt) * z + dt * x_pred
        return jnp.clip(z, -1, 1)

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        leaf_ndims = jax.tree_util.tree_map(
            lambda x: x.ndim - 1,
            ex_observations
        )
        config['leaf_ndims'] = leaf_ndims


        action_dim = ex_actions.shape[-1]
        action_len = ex_actions.shape[1]
        
        full_actions = jnp.reshape(
            ex_actions,
            (ex_actions.shape[0], -1)
        )
        full_action_dim = full_actions.shape[-1]
        ex_times = full_actions[..., :1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['latent_actor'] = encoder_module()
            # The DiT takes one token per camera, not one for all of them.
            encoders['actor_bc_flow'] = encoder_module(as_token=config['use_DiT'])

        critic_def = Value(
            hidden_dims=config['critic_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_critic'],
            encoder=encoders.get('critic'),
        )

        if config['use_DiT']:
            model_cls = mf_dit_models[config['size_DiT']]
            actor_bc_flow_def = model_cls(
                output_dim=action_dim,  
                output_len=action_len,
                encoder=encoders['actor_bc_flow'],
                use_r=True,
                residual_scale=config['residual_scale'],
            )
        else:
            actor_bc_flow_def = ActorMeanFlowField(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=full_action_dim,
                layer_norm=config['actor_layer_norm'],
                encoder=encoders.get('actor_bc_flow'),
                use_fourier_features=config['use_fourier_features'],
                fourier_feature_dim=config['fourier_feature_dim']
            )

        latent_actor_def = ActorVectorField(
            hidden_dims=config['latent_actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config.get('latent_actor_layer_norm', False),
            latent_dist='sphere_exp',
            log_std_min=_LOG_STD_MIN,
            log_std_max=config['latent_log_std_max'],
            log_std_init=_sac_log_std_init(config, full_action_dim),
            encoder=encoders.get('latent_actor'),
        )

        network_info = dict(
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, full_actions, ex_times, ex_times)),
            latent_actor=(latent_actor_def, (ex_observations,)),
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
            latent_alpha=(LogParam(), ()),
        )

        if encoders.get('actor_bc_flow') is not None:
            # Add actor_bc_flow_encoder to ModuleDict to make it separately callable.
            if isinstance(ex_observations, (dict, flax.core.FrozenDict)):
                network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations['image'],))
            else:
                network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)

        network_tx = make_optimizer(config)

        variables = network_def.init(init_rng, **network_args)
        network_params = variables['params']
        batch_stats = variables.get('batch_stats', flax.core.FrozenDict({}))
        network = TrainState.create(
            network_def, 
            network_params, 
            tx=network_tx,
            batch_stats=batch_stats
        )

        params = network.params
        params['modules_target_critic'] = params['modules_critic']


        config['action_dim'] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))



_CRITIC_RHO = 0.5
_LOG_STD_MIN = -5.0


def _target_entropy(d, angle_deg):
    """Entropy of the tangent Gaussian at the target angle, over the d-1 tangent dimensions."""
    sigma = jnp.deg2rad(angle_deg) * jnp.sqrt(d) / jnp.sqrt(d - 1.0)
    return (d - 1.0) * (jnp.log(sigma) + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e))


def _sac_log_std_init(config, d):
    """log_std starts ON target.  alpha starts at 1.0 and, without ``latent_entropy_per_dim``, the
    entropy term is d times larger per nat than the lam-normalised Q term, so any other start pins
    log_std to its ceiling for tens of thousands of steps while alpha decays."""
    return float(np.log(np.deg2rad(config['latent_target_angle_deg'])
                        * np.sqrt(d) / np.sqrt(d - 1.0)))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='lps',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            critic_hidden_dims=(256, 256, 256, 256),  # Value network hidden dimensions.
            latent_actor_hidden_dims=(256, 256),
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            latent_actor_layer_norm=False,  # Layer norm in the latent actor; bounds the scale the sphere projection discards.
            latent_entropy_per_dim=False,  # Entropy term per tangent dimension in the latent actor loss.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            num_critic=2, # critic ensemble size
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            horizon_length=ml_collections.config_dict.placeholder(int), # will be set
            use_fourier_features=False,
            fourier_feature_dim=64,
            weight_decay=0.,
            flow_ratio=0.5,
            latent_dist='sphere',
            latent_target_angle_deg=28.5,
            latent_log_std_max=0.0,
            alpha=1.0,
            use_DiT=False,
            size_DiT='base',
            residual_scale=1.0,  # Per-block residual mix of the DiT
            adaptive_reduce='mean',  # Per-sample aggregation in the adaptive L2 loss
            time_dist='uniform',  # Sampling distribution for (t, r)
            training_steps=1000000,
        )
    )
    return config
