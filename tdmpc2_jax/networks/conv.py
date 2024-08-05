from typing import Callable, Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp
import jax

from tdmpc2_jax.common.activations import mish
from tdmpc2_jax.networks.mlp import NormedLinear


class PixelPreprocess(nn.Module):
    @nn.compact
    def __call__(self, x):
        return (x / 255.0) - 0.5


class ShiftAug(nn.Module):
    """
    Random shift image augmentation.
    Adapted from https://github.com/facebookresearch/drqv2
    """

    pad: int = 3

    @nn.compact
    def __call__(self, x):
        # pad image
        c, h, w = x.shape[-3:]
        upper_shape = x.shape[:-3]
        x = jnp.reshape(x, (-1, c, h, w))
        x = jnp.pad(
            x, [(0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)], mode="edge"
        )
        shift = jax.random.randint(
            self.make_rng("augmentation"), (x.shape[0], 2), -2 * self.pad + 1, 0
        )
        return jax.vmap(lambda x, shift: jnp.roll(x, shift, axis=(1, 2))[:, :h, :w])(
            x, shift
        ).reshape(upper_shape + (c, h, w))


class PixelPreProcess(nn.Module):
    """
    Normalizes pixel observations to [-0.5, 0.5].
    """

    @nn.compact
    def __call__(self, x):
        return (x / 255.0) - 0.5


class ImageEncoder(nn.Module):
    num_channels: int
    activation: Callable[[jax.Array], jax.Array] = None
    dtype: jnp.dtype = jnp.bfloat16  # Switch this to bfloat16 for speed
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        front_shape = x.shape[:-3]
        c, h, w = x.shape[-3:]
        x = jnp.reshape(x, (-1, c, h, w))
        x.transpose(0, 2, 3, 1)
        model = nn.Sequential(
            [
                ShiftAug(),
                PixelPreprocess(),
                nn.Conv(
                    c,
                    kernel_size=(7, 7),
                    strides=(2, 2),
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                ),
                nn.relu,
                nn.Conv(
                    self.num_channels,
                    kernel_size=(5, 5),
                    strides=(2, 2),
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                ),
                nn.relu,
                nn.Conv(
                    self.num_channels,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                ),
                nn.relu,
                nn.Conv(
                    self.num_channels,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                ),
                nn.relu,
            ]
        )
        x = model(x)
        x = x.reshape(front_shape + (-1,))
        if self.activation is not None:
            x = self.activation(x)
        return x


class MultiModalEncoder(nn.Module):
    image_encoder: nn.Module
    mlp_encoder: nn.Module
    fuse_dim: int
    latent_dim: int
    activation: Callable[[jax.Array], jax.Array] = None
    dtype: jnp.dtype = jnp.bfloat16  # Switch this to bfloat16 for speed
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, im):
        x = self.mlp_encoder(x)
        im = self.image_encoder(im)
        x = jnp.concatenate([x, im], axis=-1)
        x = NormedLinear(features=self.fuse_dim, activation=mish, dtype=self.dtype)(x)
        x = NormedLinear(features=self.fuse_dim, activation=mish, dtype=self.dtype)(x)
        x = NormedLinear(
            features=self.latent_dim, activation=self.activation, dtype=self.dtype
        )(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


if __name__ == "__main__":
    import jax
    import jax.numpy as jnp
    import jax.random
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    img = jnp.array(np.random.random((16, 64, 64, 3)))
    # make right half of the image black
    img = img.at[:, :, 32:, :].set(0)
    shift_aug = ShiftAug()
    var = shift_aug.init(
        {"params": jax.random.PRNGKey(0), "augmentation": jax.random.PRNGKey(1)}, img
    )
    print(var)
    out = shift_aug.apply(var, img, rngs={"augmentation": jax.random.PRNGKey(5)})

    # show result
    fig, ax = plt.subplots(2, 1, figsize=(5, 5))
    ax[0].imshow(img[0])
    ax[1].imshow(out[0])
    plt.show()
