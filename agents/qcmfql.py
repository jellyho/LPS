import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from agents.meanflow_utils import adaptive_l2_loss, sample_t_r, sample_latent_dist
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field, get_batch_shape
from utils.networks import ActorMeanFlowField, Value, ActorVectorField
from utils.dit import mf_dit_models


class QCMFQLAgent(flax.struct.PyTreeNode):
    """Don't extract but select! with action chunking. 
    """
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the behavior policy evaluation loss"""
        # Q(s, a) <- R + \gamma V(s')
        batch_size = batch['actions'].shape[0]

        rng, sample_rng = jax.random.split(rng, 2)
        # onestep generation
        next_actions = self.sample_actions(batch['next_observations'], rng=sample_rng)

        next_qs = self.network.select('target_critic')(
            batch['next_observations'], 
            actions=next_actions.reshape(batch_size, -1)
        )

        if self.config['num_critic'] > 1:
            if self.config['critic_agg'] == 'min':
                next_q = jnp.min(next_qs, axis=0)
            else:
                next_q = jnp.mean(next_qs, axis=0)
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
        """Compute the FQL actor loss."""
        batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))  # fold in horizon_length together with action_dim
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng, drop_rng = jax.random.split(rng, 4)

        # BC mean flow loss.
        x = batch_actions
        t, r = sample_t_r(batch_size, t_rng, self.config['flow_ratio'])

        ##### It does not need to be normal distirbution
        e = sample_latent_dist(x_rng, (batch_size, action_dim), self.config['latent_dist'])
        z = (1 - t) * x + t * e
        v = e - x

        def mean_flow_forward(z, t, r):
            # Call following the network input order (Obs, Z, T, R)
            return self.network.select('actor_bc_flow')(
                batch['observations'], 
                z, 
                t, 
                t - r, # This seems to work better
                params=grad_params
            )

        v = e - x
        x_pred, dxdt = jax.jvp(
            mean_flow_forward, 
            (z, t, r), 
            (v, jnp.ones_like(t), jnp.zeros_like(r))
        )
        u, dudt = z - x_pred, v - dxdt
        u_tgt = v - jnp.clip(t - r, a_min=0.0, a_max=1.0) * dudt
        u_tgt = jax.lax.stop_gradient(u_tgt)
        err = u - u_tgt

        loss = adaptive_l2_loss(err)

        return loss, {
            'actor_loss': loss,
            'mf/u_mean': u.mean(),
            'mf/v_mean': v.mean(),
            'mf/dudt_mean': dudt.mean(),
        }

    def onestep_actor_loss(self, batch, grad_params, rng):
        observations = batch['observations']
        batch_size = batch['actions'].shape[0]
        latent_dim = self.config['action_dim'] * self.config['horizon_length']
        
        ### Query latent actor
        rng, x_rng, l_rng = jax.random.split(rng, 3)
        # normal = jax.random.normal(x_rng, (batch_size, latent_dim))
        e = sample_latent_dist(x_rng, (batch_size, latent_dim), self.config['latent_dist'])

        onestep_pred = self.network.select('onestep_actor')(
            observations, 
            e,
            rng=l_rng,
            params=grad_params # <--- Gradients flow here
        )

        rng, x_rng = jax.random.split(rng, 2)
        x_pred = self.compute_flow_actions(observations, e)
        a_pred_flat = jnp.reshape(x_pred, (batch_size, latent_dim))

        info_dict = {}

        qs = self.network.select('critic')(
            observations,
            onestep_pred.reshape(batch_size, -1)
        )

        if self.config['num_critic'] > 1:
            q = jnp.mean(qs, axis=0)
        else:
            q = qs

        q_loss = -q.mean()
        lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
        q_loss = lam * q_loss

        mse_loss = jnp.mean(jnp.square(a_pred_flat - onestep_pred))
        loss = q_loss + mse_loss * self.config['alpha']
            
        info_dict['onestep_loss'] = loss
        info_dict['mse_loss'] = mse_loss
        info_dict['q_loss'] = q_loss

        return loss, info_dict

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng, value_rng = jax.random.split(rng, 4)

        loss = 0

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        loss += critic_loss
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        loss += actor_loss
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        onestep_loss, onestep_info = self.onestep_actor_loss(batch, grad_params, actor_rng)
        loss += onestep_loss
        for k, v in onestep_info.items():
            info[f'onestep/{k}'] = v

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
    def sample_values(
        self,
        observations,
        rng=None,
    ):
        return jnp.zeros_like(observations)
    
    @jax.jit
    def sample_actions(
        self,
        observations,
        rng=None,
    ):
        """
        Sample actions. 
        Note: CFG logic and step size are handled inside 'compute_flow_actions' using self.config.
        """
        latent_dim = self.config["horizon_length"] * self.config["action_dim"]
        batch_shape = get_batch_shape(observations, self.config['leaf_ndims'])

        rng, x_rng = jax.random.split(rng, 2)
        e = sample_latent_dist(x_rng, (*batch_shape, latent_dim), self.config['latent_dist'])
        
        actions = self.network.select('onestep_actor')(
            observations, 
            e,
            is_encoded=False
        )
        actions = jnp.clip(actions, -1, 1)
        actions = jnp.reshape(
            actions, 
            (*batch_shape,  # batch_size
            self.config["horizon_length"], self.config["action_dim"])
        )
            
        return actions

    @jax.jit
    def compute_flow_actions(
        self,
        observation,
        noise,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        t = jnp.ones_like((noise[..., :1]))
        r = jnp.zeros_like((noise[..., :1]))
        
        output = self.network.select('actor_bc_flow')(
            observation,
            noise, 
            t, 
            t - r,
            is_encoded=False
        )
        action = jnp.clip(output, -1, 1)
        return action

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
            encoders['onestep_actor'] = encoder_module()

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
                use_r=True
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

        onestep_actor_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('onestep_actor'),
        )

        network_info = dict(
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, full_actions, ex_times, ex_times)),
            onestep_actor=(onestep_actor_def, (ex_observations, full_actions)),
            critic=(critic_def, (ex_observations, full_actions)),
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
                    init_value=0.0,             # Initial LR at warmup start (usually 0)
                    peak_value=config['lr'],    # Max LR reached after warmup
                    warmup_steps=warmup_steps,
                    decay_steps=decay_steps, # Total decay period
                    end_value=config['lr'] / 10 # LR at end of training (usually peak/10 ~ 0)
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
            agent_name='qcmfql',  # Agent name.
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
            critic_agg='min',  # Aggregation method for target Q values.
            num_critic=2, # critic ensemble size
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            horizon_length=ml_collections.config_dict.placeholder(int), # will be set
            use_fourier_features=False,
            fourier_feature_dim=64,
            weight_decay=0.,
            latent_dist='normal',
            alpha=1.0,
            flow_ratio=0.25,
            training_steps=10000,
            use_DiT=False,
            size_DiT='base',
            
            ####### Unused parameters for compatibility #######
            extract_method="unused",
            mf_method="unused"
        )
    )
    return config
