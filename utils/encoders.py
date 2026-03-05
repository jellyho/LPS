import functools
from typing import Sequence, Any

import flax.linen as nn
import jax.numpy as jnp

from utils.networks import MLP

import jax
import flaxmodels  # pip install flaxmodels
import numpy as np


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


class ImpalaEncoder(nn.Module):
    """IMPALA encoder."""

    width: int = 1
    stack_sizes: tuple = (16, 32, 32)
    num_blocks: int = 2
    dropout_rate: float = None
    mlp_hidden_dims: Sequence[int] = (512,)
    layer_norm: bool = False

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
        out = conv_out.reshape((*x.shape[:-3], -1))

        out = MLP(self.mlp_hidden_dims, activate_final=True, layer_norm=self.layer_norm)(out)

        return out

class RobotResNet(nn.Module):
    """
    A wrapper around flaxmodels.ResNet18/50.
    1. Loads a pretrained backbone.
    2. Extracts the final feature map.
    3. Applies SpatialSoftmax.
    4. Projects to a user-defined embedding size.
    """
    backbone_name: str = 'resnet18' # 'resnet18' or 'resnet50'
    use_spatial_softmax: bool = True
    freeze_backbone: bool = True    # If True, stops gradients from flowing into ResNet

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
        features = list(features_dict.values())[-2]
        # backbone_train_mode = train and not self.freeze_backbone
        
        # Get dictionary of feature maps
        # features_dict = backbone(x, train=backbone_train_mode)
        
        # Extract the last spatial feature map.
        # For ResNet18/34 in flaxmodels, the last block is usually 'block4_1'.
        # For ResNet50, it is usually 'block4_2'. 
        # To be safe, we grab the last value from the dictionary.
        # features = list(features_dict.values())[-2]
        
        # [Optional] Stop Gradient to freeze weights strictly
        if self.freeze_backbone:
            features = jax.lax.stop_gradient(features)

        # features = nn.Dense(self.output_dim)(features)
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

    @nn.compact
    def __call__(self, x, train: bool = True):
        # 1. Parse Input Shape
        # x shape: (Batch, Height, Width, Channels)

        # is_unbatched = (x.ndim == 3)
        
        # if is_unbatched:
        #     # (H, W, C) -> (1, H, W, C)
        #     x = jnp.expand_dims(x, axis=0)

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
            encoded = self.encoder_cls(name=f'view_{i}')(view, train=train)
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
# 3. Main Execution / Testing
# -----------------------------------------------------------------------------

base_modules = {
    'impala': ImpalaEncoder,
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