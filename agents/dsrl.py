import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import ml_collections

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field, get_batch_shape, make_optimizer
from utils.networks import ActorVectorField, Value, LogParam
from utils.dit import mf_dit_models

class DSRLAgent(flax.struct.PyTreeNode):
    """DSRL agent with action chunking.

    A latent actor steers a BC flow policy through its noise space. It climbs a noise-space critic,
    which is regressed onto the action critic's value of the decoded action.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the critic TD loss."""
        batch_size = batch['actions'].shape[0]
        batch_actions = jnp.reshape(batch["actions"], (batch_size, -1))
        
        # TD loss
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
            actions=batch_actions, 
            params=grad_params
        )

        target_q = batch['rewards'] + \
            (self.config['discount'] ** self.config["horizon_length"]) * batch['masks'] * next_q

        
        critic_loss = (jnp.square(qs - jnp.broadcast_to(target_q, qs.shape))).mean()

        q = qs.mean(axis=0) # For logging

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def sample_latent_dist(self, x_rng, sample_shape):
        if self.config['latent_dist'] == 'normal':
            e = jax.random.normal(x_rng, sample_shape)
        elif self.config['latent_dist'] == 'uniform':
            e = jax.random.uniform(x_rng, sample_shape, minval=-1.0, maxval=1.0)
        elif self.config['latent_dist'] == 'sphere':
            e = jax.random.normal(x_rng, sample_shape)
            sq_sum = jnp.sum(jnp.square(e), axis=-1, keepdims=True)
            norm = jnp.sqrt(sq_sum + 1e-6)
            e = e / norm * jnp.sqrt(sample_shape[-1])
        return e

    def actor_loss(self, batch, grad_params, rng):
        """Compute the BC flow-matching loss."""
        batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))  # fold in horizon_length together with action_dim
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        x_0 = self.sample_latent_dist(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc_flow')(
            batch['observations'], 
            x_t, 
            t, 
            params=grad_params
        )

        bc_flow_loss = jnp.mean(jnp.square(pred - vel))
        # Total loss.
        actor_loss = bc_flow_loss

        return actor_loss, {
            'actor_loss': actor_loss,
        }
    
    @jax.jit
    def sample_values(
        self,
        observations,
        rng=None,
    ):
        return jnp.zeros_like(observations)
    
    def latent_actor_loss(self, batch, grad_params, rng):
        observations = batch['observations']
        batch_size = batch['actions'].shape[0]
        latent_dim = self.config['action_dim'] * self.config['horizon_length']
        
        ### Query latent actor
        rng, x_rng, l_rng = jax.random.split(rng, 3)
        rng, e_rng = jax.random.split(rng, 2)
        _eps = jax.random.normal(e_rng, (batch_size, latent_dim))
        z_pred, sac_logp, sac_log_std = self.network.select('latent_actor')(
            observations,
            v_base=_eps,
            rng=l_rng,
            return_logp=True,
            params=grad_params
        )
        # Noise-space critic Q(s, z).
        qs = self.network.select('noise_critic')(
            observations,
            z_pred.reshape(batch_size, -1)
        )

        info_dict = {
            'z_norm': jnp.mean(jnp.square(z_pred)),
        }

        if self.config['num_critic'] > 1:
            q = jnp.mean(qs, axis=0)  # The latent actor climbs the ensemble mean.
        else:
            q = qs

        lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
        alpha_sac = self.network.select('latent_alpha')(params=grad_params)
        q_loss = -q.mean() + jax.lax.stop_gradient(alpha_sac) * sac_logp.mean()
        q_loss = self.config['alpha'] * lam * q_loss
        ent = -jax.lax.stop_gradient(sac_logp).mean()
        tgt = _target_entropy(latent_dim)
        q_loss = q_loss + (alpha_sac * (ent - tgt)).mean()
        info_dict['sac_entropy'] = ent
        info_dict['sac_entropy_target'] = jnp.asarray(tgt, dtype=ent.dtype)
        info_dict['sac_alpha'] = alpha_sac
        info_dict['sac_log_std'] = sac_log_std.mean()
        _m = 0.05 * (_LOG_STD_MAX - _LOG_STD_MIN)
        info_dict['sac_frac_at_clip'] = (
            (sac_log_std <= _LOG_STD_MIN + _m) | (sac_log_std >= _LOG_STD_MAX - _m)).mean()
        info_dict['latent_loss'] = q_loss

        # Noise aliasing: fit the noise critic to Q(s, decode(e)) for prior noise e.
        e = self.sample_latent_dist(x_rng, (batch_size, latent_dim))
        x_pred = self.compute_flow_actions(observations, e)

        noise_qs = self.network.select('noise_critic')(
            observations,
            e,
            params=grad_params
        )
        target_qs = self.network.select('critic')(
            observations,
            x_pred,
        )

        if self.config['num_critic'] > 1:
            target_q = jnp.mean(target_qs, axis=0)
        else:
            target_q = target_qs

        alias_loss = jnp.mean(jnp.square((noise_qs - target_q)))
        info_dict['alias_loss'] = alias_loss
        loss = q_loss + alias_loss

        return loss, info_dict

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng, latent_rng = jax.random.split(rng, 4)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        latent_loss, latent_info = self.latent_actor_loss(batch, grad_params, latent_rng)
        for k, v in latent_info.items():
            info[f'latent/{k}'] = v

        loss = critic_loss + actor_loss + latent_loss
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
    def sample_actions(
        self,
        observations,
        rng=None,
    ):
        latent_dim = self.config["horizon_length"] * self.config["action_dim"]
        batch_shape = get_batch_shape(observations, self.config['leaf_ndims'])

        rng, x_rng = jax.random.split(rng, 2)
        rng, s_rng = jax.random.split(rng, 2)
        _e = jax.random.normal(s_rng, (*batch_shape, latent_dim))
        noises = self.network.select('latent_actor')(observations, v_base=_e)
        actions = self.compute_flow_actions(observations, noises)
        actions = jnp.reshape(
            actions, 
            (*batch_shape, self.config["horizon_length"], self.config["action_dim"])
        )
        actions = jnp.clip(actions, -1, 1)
        return actions

    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        if self.config['encoder'] is not None:
            if isinstance(observations, (dict, flax.core.FrozenDict)):
                observations['image'] = self.network.select('actor_bc_flow_encoder')(observations['image'])
            else:
                observations = self.network.select('actor_bc_flow_encoder')(observations)

        actions = noises
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full(noises[..., :1].shape, i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions

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
            encoders['actor_bc_flow'] = encoder_module()
            encoders['latent_actor'] = encoder_module()

        # Define networks.
        critic_def = Value(
            hidden_dims=config['critic_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_critic'],
            encoder=encoders.get('critic'),
        )

        noise_critic_def = Value(
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
                use_r=False
            )
        else:
            actor_bc_flow_def = ActorVectorField(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=full_action_dim,
                layer_norm=config['actor_layer_norm'],
                encoder=encoders.get('actor_bc_flow'),
            )

        latent_actor_def = ActorVectorField(
            # The latent (steering) actor has its own width, separate from the base flow's.
            hidden_dims=config['latent_actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('latent_actor'),
            latent_dist='sphere_exp',
            log_std_min=_LOG_STD_MIN,
            log_std_max=_LOG_STD_MAX,
            log_std_init=_sac_log_std_init(config, full_action_dim),
        )

        network_info = dict(
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, full_actions, ex_times)),
            latent_actor=(latent_actor_def, (ex_observations,)),
            critic=(critic_def, (ex_observations, full_actions)),
            noise_critic=(noise_critic_def, (ex_observations, full_actions)),
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
_TARGET_ANGLE_DEG = 28.5
_LOG_STD_MIN, _LOG_STD_MAX = -5.0, 0.0


def _target_entropy(d):
    """Entropy of the tangent Gaussian at the target angle, over the d-1 tangent dimensions."""
    sigma = jnp.deg2rad(_TARGET_ANGLE_DEG) * jnp.sqrt(d) / jnp.sqrt(d - 1.0)
    return (d - 1.0) * (jnp.log(sigma) + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e))


def _sac_log_std_init(config, d):
    return float(np.log(np.deg2rad(_TARGET_ANGLE_DEG) * np.sqrt(d) / np.sqrt(d - 1.0)))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='dsrl',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            critic_hidden_dims=(256, 256, 256, 256),  # Value network hidden dimensions.
            latent_actor_hidden_dims=(256, 256),
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            alpha=1.0,  # Scale of the latent actor's Q objective.
            num_critic=2, # critic ensemble size
            flow_steps=10,  # Number of flow steps.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            horizon_length=ml_collections.config_dict.placeholder(int), # will be set
            use_fourier_features=False,
            fourier_feature_dim=64,
            weight_decay=0.,
            latent_dist='sphere',
            use_DiT=False,
            size_DiT='base',
            training_steps=1000000,
        )
    )
    return config
