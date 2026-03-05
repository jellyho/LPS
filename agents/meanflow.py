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
from utils.networks import ActorMeanFlowField
from utils.dit import mf_dit_models

class MEANFLOWAgent(flax.struct.PyTreeNode):
    """Don't extract but select! with action chunking. 
    """
    rng: Any
    network: Any
    config: Any = nonpytree_field()
        
    @jax.jit
    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))  # fold in horizon_length together with action_dim
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC mean flow loss.
        x = batch_actions
        t, r = sample_t_r(batch_size, t_rng, self.config['flow_ratio'])

        ##### It does not need to be normal distirbution
        e = sample_latent_dist(x_rng, (batch_size, action_dim), self.config['latent_dist'])

        if self.config['mf_method'] in ['jit_mf', 'mfql']:
            z = (1 - t) * x + t * e
            v = e - x
        else:
            z = r * x + (1 - r) * e
            v = x - e

        def mean_flow_forward(z, t, r):
            # Call following the network input order (Obs, Z, T, R)
            return self.network.select('actor_bc_flow')(
                batch['observations'], 
                z, 
                t, 
                t - r, # This seems to work better
                params=grad_params
            )
        def mean_flow_jit_forward(z, t, r):
            # Call following the network input order (Obs, Z, T, R)
            x_pred = self.network.select('actor_bc_flow')(
                batch['observations'], 
                z, 
                t - r, 
                r,
                params=grad_params
            )
            u = (x_pred - z) / (1 - r)
            return u

        ############################################
        if self.config['mf_method'] == 'jit_mf':
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
        ############################################
        elif self.config['mf_method'] == 'mfql':
            g_pred, dgdt = jax.jvp(
                mean_flow_forward, 
                (z, t, r), 
                (v, jnp.ones_like(t), jnp.zeros_like(r))
            )
            g_tgt = z + (t - r - 1) * v - (t - r) * dgdt
            g_tgt = jax.lax.stop_gradient(g_tgt)
            err = g_pred - g_tgt
            loss = adaptive_l2_loss(err)
        elif self.config['mf_method'] == 'jit':
            u_pred, dudr = jax.jvp(
                mean_flow_jit_forward, 
                (z, t, r), 
                (v, jnp.zeros_like(t), jnp.ones_like(r))
            )
            u_tgt = v + jnp.clip(t - r, a_min=0.0, a_max=1.0) * dudr
            u_tgt = jax.lax.stop_gradient(u_tgt)
            err = u_pred - u_tgt
            loss = adaptive_l2_loss(err)
        else:
            raise NotImplementedError
        #########################################

        return loss, {
            'actor_loss': loss,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng = jax.random.split(rng, 2)

        loss = 0
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        loss += actor_loss
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        return loss, info

    @staticmethod
    def _update(agent, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)
    
    @jax.jit
    def batch_update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
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
        rng, x_rng = jax.random.split(rng, 2)
        latent_dim = self.config["horizon_length"] * self.config["action_dim"]
        batch_shape = get_batch_shape(observations, self.config['leaf_ndims'])

        e = sample_latent_dist(
            x_rng, 
            (*batch_shape, latent_dim), 
            self.config['latent_dist']
        )
        actions = self.compute_flow_actions(observations, e)
        actions = jnp.reshape(
            actions, 
            (*batch_shape, self.config["horizon_length"], self.config["action_dim"])
        )
        return actions
    
    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def compute_flow_actions(
        self,
        observation,
        noise,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        t = jnp.ones_like((noise[..., :1]))
        r = jnp.zeros_like((noise[..., :1]))

        if self.config['mf_method'] == 'jit':
            output = self.network.select('actor_bc_flow')(
                observation,
                noise, 
                t - r, 
                r,
            )
        else:
            output = self.network.select('actor_bc_flow')(
                observation,
                noise, 
                t, 
                t - r
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
            encoders['actor_bc_flow'] = encoder_module()

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

        actor_input_shape = (ex_observations, full_actions, ex_times, ex_times)

        network_info = dict(
            actor_bc_flow=(actor_bc_flow_def, actor_input_shape),
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
        config['action_dim'] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='meanflow',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
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
            flow_ratio=0.25,
            latent_dist='normal',
            use_DiT=False,
            size_DiT='base',
            mf_method='jit_mf',
            training_steps=1000000,

            ######### unused
            alpha=1.0,
            extract_method='unused',
            critic_hidden_dims='unused',
            latent_actor_hidden_dims='unused'
        )
    )
    return config
