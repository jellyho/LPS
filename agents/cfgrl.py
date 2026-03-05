import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field, get_batch_shape
from utils.networks import ActorVectorField, Value
from utils.dit import mf_dit_models

class CFGRLAgent(flax.struct.PyTreeNode):
    """Flow Q-learning (FQL) agent with action chunking. 
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the FQL critic loss."""
        batch_size = batch['actions'].shape[0]
        batch_actions = jnp.reshape(batch["actions"], (batch_size, -1))
        
        # TD loss
        rng, sample_rng = jax.random.split(rng, 2)
        next_v = self.network.select('value')(
            batch['next_observations'], 
        )

        target_q = batch['rewards'] + \
            (self.config['discount'] ** self.config["horizon_length"]) * batch['masks'] * next_v
        
        qs = self.network.select('critic')(
            batch['observations'], 
            actions=batch_actions, 
            params=grad_params
        )
        
        critic_loss = (jnp.square(qs - jnp.broadcast_to(target_q, qs.shape))).mean()
        q = qs.mean(axis=0) # For logging

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def value_loss(self, batch, grad_params, rng):
        batch_size = batch['actions'].shape[0]
        batch_actions = jnp.reshape(batch["actions"], (batch_size, -1))

        qs = self.network.select('target_critic')(
            batch['observations'],
            batch_actions
        )

        q = jnp.min(qs, axis=0)

        v = self.network.select('value')(
            batch['observations'],
            params=grad_params
        )

        value_loss = self.expectile_loss(q - v, self.config['expectile']).mean()

        return value_loss, {
            'value_loss': value_loss,
            'v': v.mean(),
            'v_min' : v.min(),
            'v_max' : v.max()
        }

    def expectile_loss(self, diff, expectile=0.9):
        weight = jnp.where(diff > 0, expectile, (1 - expectile))
        return weight * (diff**2)

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
        """Compute the FQL actor loss."""
        batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))  # fold in horizon_length together with action_dim
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        v = self.network.select('value')(
            batch['observations']
        )
        qs = self.network.select('target_critic')(
            batch['observations'],
            batch_actions
        )
        q = jnp.min(qs, axis=0)
        exp_a = jnp.exp((q - v) * self.config['temperature'])
        exp_a = jnp.minimum(exp_a, 100.0)
        exp_a = ((q-v) > 0).astype(jnp.float32)

        # BC flow loss.
        x_0 = self.sample_latent_dist(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        idx_positive = jnp.ones((x_1.shape[0], 1))
        pred_positive = self.network.select('actor_bc_flow')(
            batch['observations'], 
            x_t, 
            t,
            g=idx_positive, 
            params=grad_params
        )
        loss_positive = jnp.mean(jnp.square(pred_positive - vel), axis=-1) * exp_a

        idx_uncond = jnp.zeros((x_1.shape[0], 1))
        pred_uncond = self.network.select('actor_bc_flow')(
            batch['observations'], 
            x_t, 
            t,
            g=idx_uncond, 
            params=grad_params
        )
        loss_uncond = jnp.mean(jnp.square(pred_uncond - vel), axis=-1)

        actor_loss = jnp.mean(loss_positive + 0.1 * loss_uncond)

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

        value_loss, value_info = self.value_loss(batch, grad_params, latent_rng)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        loss = critic_loss + actor_loss + value_loss
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
    def batch_update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        # update_size = batch["observations"].shape[0]
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)
    
    @jax.jit
    def sample_actions(
        self,
        observations,
        rng=None,
    ):
        latent_dim = self.config["horizon_length"] * self.config["action_dim"]
        batch_shape = get_batch_shape(observations, self.config['leaf_ndims'])

        rng, x_rng = jax.random.split(rng, 2)
        noises = self.sample_latent_dist(x_rng, (*batch_shape, latent_dim))
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
        idx_positive = jnp.ones(noises[..., :1].shape)
        idx_uncond = jnp.zeros(noises[..., :1].shape)
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full(noises[..., :1].shape, i / self.config['flow_steps'])
            vels_positive = self.network.select('actor_bc_flow')(
                observations, 
                actions, 
                t, 
                g=idx_positive, 
                is_encoded=True
            )
            vels_uncond = self.network.select('actor_bc_flow')(
                observations, 
                actions, 
                t, 
                g=idx_uncond,
                is_encoded=True
            )
            vels = vels_uncond + self.config['cfg'] * (vels_positive - vels_uncond)
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

        value_def = Value(
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

        network_info = dict(
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, full_actions, ex_times, ex_times)),
            critic=(critic_def, (ex_observations, full_actions)),
            value=(value_def, (ex_observations,)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
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

        if config["weight_decay"] > 0.:
            network_tx = optax.adamw(learning_rate=config['lr'], weight_decay=config["weight_decay"])
        else:
            if config['use_DiT']:
                warmup_steps = int(config['training_steps'] * 0.01)
                decay_steps = config['training_steps'] - warmup_steps
                lr_schedule = optax.warmup_cosine_decay_schedule(
                    init_value=0.0,             # Initial LR (usually 0)
                    peak_value=config['lr'],    # Peak LR after warmup
                    warmup_steps=warmup_steps,
                    decay_steps=decay_steps, # Total decay duration
                    end_value=config['lr'] / 10 # Final LR (usually 1/10 of peak ~ 0)
                )
                network_tx = optax.chain(
                    optax.clip_by_global_norm(1.0),
                    optax.adam(learning_rate=lr_schedule),
                )
            else:
                network_tx = optax.adam(learning_rate=config['lr'])

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
        params[f'modules_target_critic'] = params[f'modules_critic']

        # config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='cfgrl',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            critic_hidden_dims=(256, 256, 256, 256),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            critic_agg='mean',  # Aggregation method for target Q values.
            alpha=1.0,  # BC coefficient (need to be tuned for each environment).
            num_critic=2, # critic ensemble size
            flow_steps=10,  # Number of flow steps.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            horizon_length=ml_collections.config_dict.placeholder(int), # will be set
            use_fourier_features=False,
            fourier_feature_dim=64,
            weight_decay=0.,
            latent_dist='normal',

            use_DiT=False,
            size_DiT='base',
            training_steps=10000,

            expectile=0.9,      
            temperature=3.0,    
            cfg=1.5,            

            ## unsued
            mf_method='unused',
            extract_method='unused', # 'ddpg', 'awr',,
            latent_actor_hidden_dims=(256, 256),
        )
    )
    return config
