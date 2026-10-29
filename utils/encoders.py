import functools
from typing import Sequence, Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from utils.networks import MLP

import flaxmodels  # pip install flaxmodels


class ResnetStack(nn.Module):
    """ResNet stack module."""

    num_features: int
    num_blocks: int
    max_pooling: bool = True

    @nn.compact
    def __call__(self, x):
        initializer = nn.initializers.xavier_uniform()
        conv_out = nn.Conv(
            features=self.num_features,
            kernel_size=(3, 3),
            strides=1,
            kernel_init=initializer,
            padding='SAME',
        )(x)

        if self.max_pooling:
            conv_out = nn.max_pool(
                conv_out,
                window_shape=(3, 3),
                padding='SAME',
                strides=(2, 2),
            )

        for _ in range(self.num_blocks):
            block_input = conv_out
            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
            )(conv_out)

            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
            )(conv_out)
            conv_out += block_input

        return conv_out


def _spatial_softmax(feat):
    """Expected keypoint coordinates, two numbers per channel.

    Flattening the conv map instead would put the whole spatial grid into the first dense layer --
    28x28x32 is 25,088 inputs, and that one layer would hold 99% of the encoder's parameters. A softmax
    over the grid keeps where each channel fires, which is what matters for manipulation, and
    leaves the dense layer with two numbers per channel.
    """
    *batch, h, w, c = feat.shape
    flat = feat.reshape(*batch, h * w, c)
    weights = jax.nn.softmax(flat, axis=-2)

    ys = jnp.linspace(-1.0, 1.0, h)
    xs = jnp.linspace(-1.0, 1.0, w)
    grid_y, grid_x = jnp.meshgrid(ys, xs, indexing='ij')
    coords = jnp.stack([grid_y.reshape(-1), grid_x.reshape(-1)], axis=-1)  # (h*w, 2)

    keypoints = jnp.einsum('...nc,nd->...cd', weights, coords)             # (..., c, 2)
    return keypoints.reshape(*batch, c * 2)


class ImpalaEncoder(nn.Module):
    """IMPALA encoder."""

    width: int = 1
    stack_sizes: tuple = (16, 32, 32)
    num_blocks: int = 2
    dropout_rate: float = None
    mlp_hidden_dims: Sequence[int] = (512,)
    layer_norm: bool = False
    as_token: bool = False
    readout: str = 'softmax'      # 'softmax' | 'flatten'

    def setup(self):
        stack_sizes = self.stack_sizes
        self.stack_blocks = [
            ResnetStack(
                num_features=stack_sizes[i] * self.width,
                num_blocks=self.num_blocks,
            )
            for i in range(len(stack_sizes))
        ]
        if self.dropout_rate is not None:
            self.dropout = nn.Dropout(rate=self.dropout_rate)

    @nn.compact
    def __call__(self, x, train=True, cond_var=None):
        x = x.astype(jnp.float32) / 255.0

        conv_out = x

        for idx in range(len(self.stack_blocks)):
            conv_out = self.stack_blocks[idx](conv_out)
            if self.dropout_rate is not None:
                conv_out = self.dropout(conv_out, deterministic=not train)

        conv_out = nn.relu(conv_out)
        if self.layer_norm:
            conv_out = nn.LayerNorm()(conv_out)

        if self.readout == 'flatten':
            out = conv_out.reshape((*x.shape[:-3], -1))
        else:
            out = _spatial_softmax(conv_out)

        out = MLP(self.mlp_hidden_dims, activate_final=True, layer_norm=self.layer_norm)(out)

        # One token per view, for an encoder feeding a transformer.
        return out[..., None, :] if self.as_token else out

class RobotResNet(nn.Module):
    """ImageNet-pretrained flaxmodels ResNet18/50, returning its last feature map as tokens.

    The output is (B, L, D): one token per spatial location of the second-to-last activation.
    """
    backbone_name: str = 'resnet18' # 'resnet18' or 'resnet50'
    use_spatial_softmax: bool = True
    freeze_backbone: bool = True    # If True, stops gradients from flowing into ResNet
    as_token: bool = False          # already returns tokens; accepted so the wrapper can pass it

    @nn.compact
    def __call__(self, x, train: bool = True):
        # 1. Select and Initialize Backbone
        # 'pretrained="imagenet"' ensures weights are loaded during init
        if self.backbone_name == 'resnet18':
            backbone = flaxmodels.ResNet18(output='activations', pretrained='imagenet')
        elif self.backbone_name == 'resnet50':
            backbone = flaxmodels.ResNet50(output='activations', pretrained='imagenet')
        else:
            raise ValueError("Supported backbones: resnet18, resnet50")

        # 2. Forward Pass (Backbone)
        # Use train=False if freezing to keep Batch Norm statistics stable
        features_dict = backbone(x, train=False)
        # The second-to-last activation is the final spatial feature map, before pooling.
        features = list(features_dict.values())[-2]

        if self.freeze_backbone:
            features = jax.lax.stop_gradient(features)

        feature_dim = features.shape[-1]
        if len(features.shape) == 3:
            features = features.reshape(-1, feature_dim)
        else:
            batch_size = features.shape[0]
            features = features.reshape(batch_size, -1, feature_dim)        
        return features


class MultiViewWrapper(nn.Module):
    """
    Adapts an encoder to handle multiple images stacked channel-wise.
    Input: (B, H, W, C) where C is a multiple of 3.
    Output: Concatenated embeddings from N encoders.
    
    Structure:
    1. Splits input into N inputs of (B, H, W, 3).
    2. Passes each through a separate encoder instance (e.g., view_0_encoder, view_1_encoder).
    3. Concatenates the outputs.
       - If output is 1D (B, D) -> (B, N*D) (Impala style)
       - If output is sequence (B, L, D) -> (B, N*L, D) (ResNet/Transformer style)
    """
    encoder_cls: Any  # Partial function of the base encoder (e.g., ResNet, Impala)
    as_token: bool = False

    @nn.compact
    def __call__(self, x, train: bool = True):
        # 1. Parse Input Shape
        # x shape: (Batch, Height, Width, Channels)
        C = x.shape[-1]
        
        assert C % 3 == 0, f"Input channels ({C}) must be divisible by 3 (RGB)."
        num_views = C // 3
        
        # 2. Split into individual images
        # views will be a list of N tensors, each (B, H, W, 3)
        views = jnp.split(x, num_views, axis=-1)
        
        outputs = []
        for i, view in enumerate(views):
            # 3. Create & Apply Encoder
            # name=f'view_{i}' ensures separate parameters for each camera view.
            # If you want shared weights (Siamese), remove the name argument or define it outside.
            encoded = self.encoder_cls(name=f'view_{i}', as_token=self.as_token)(view, train=train)
            outputs.append(encoded)
            
        # 4. Concatenate Outputs
        # Check rank to decide concatenation axis
        if outputs[0].ndim == 2:
            # Case: Impala (B, D) -> Concat to (B, N*D)
            return jnp.concatenate(outputs, axis=-1)
        elif outputs[0].ndim == 3:
            # Case: ResNet/ViT tokens (B, L, D) -> Concat to (B, N*L, D)
            # e.g., (B, 49, D) + (B, 49, D) -> (B, 98, D)
            return jnp.concatenate(outputs, axis=1)
        else:
            # Default fallback: Concat along last dimension
            return jnp.concatenate(outputs, axis=-1)

# -----------------------------------------------------------------------------
# Encoder registry
# -----------------------------------------------------------------------------

base_modules = {
    'impala': ImpalaEncoder,
    'impala_flatten': functools.partial(ImpalaEncoder, readout='flatten'),
    'impala_debug': functools.partial(ImpalaEncoder, num_blocks=1, stack_sizes=(4, 4)),
    'impala_small': functools.partial(ImpalaEncoder, num_blocks=1),
    'impala_large': functools.partial(ImpalaEncoder, stack_sizes=(64, 128, 128), mlp_hidden_dims=(1024,)),
    'resnet18_freeze': functools.partial(RobotResNet, backbone_name='resnet18', freeze_backbone=True),
    'resnet18': functools.partial(RobotResNet, backbone_name='resnet18', freeze_backbone=False),
}

## Multi-View Wrapped
encoder_modules = {
    k: functools.partial(MultiViewWrapper, encoder_cls=v)
    for k, v in base_modules.items()
}