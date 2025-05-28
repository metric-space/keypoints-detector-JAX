import os
import sys
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp


class UpBlock2D(eqx.Module):
    layers: eqx.nn.Sequential

    def __init__(
        self, key, in_features, out_features, kernel_size=3, padding=1, groups=1
    ):
        self.layers = eqx.nn.Sequential(
            [
                eqx.nn.Conv2d(
                    key=key,
                    kernel_size=kernel_size,
                    in_channels=in_features,
                    out_channels=out_features,
                    padding=padding,
                    groups=groups,
                ),
                eqx.nn.BatchNorm(input_size=out_features, axis_name="batch"),
                eqx.nn.Lambda(jax.nn.relu),
            ]
        )

    def __call__(self, x, state):

        C, W, H = x.shape

        scale = 2
        out = jax.image.resize(x, shape=(C, scale * W, scale * H), method="nearest")
        return self.layers(out, state)


class DownBlock2D(eqx.Module):
    layers: eqx.nn.Sequential

    def __init__(
        self, key, in_features, out_features, kernel_size=3, padding=1, groups=1
    ):
        self.layers = eqx.nn.Sequential(
            [
                eqx.nn.Conv2d(
                    key=key,
                    kernel_size=kernel_size,
                    in_channels=in_features,
                    out_channels=out_features,
                    padding=padding,
                    groups=groups,
                ),
                eqx.nn.BatchNorm(input_size=out_features, axis_name="batch"),
                eqx.nn.Lambda(jax.nn.relu),
                eqx.nn.AvgPool2d(
                    kernel_size=2, stride=2
                ),  # check if the authors are overdoing it with (2,2)
            ]
        )

    def __call__(self, x, state):

        return self.layers(x, state)


class Encoder(eqx.Module):
    layers: eqx.nn.Sequential

    def __init__(
        self, key, block_expansion, in_features, num_blocks=3, max_features=256
    ):

        keys = jax.random.split(key, num_blocks)

        layers_params = [
            {
                "key": keys[i],
                "in_features": (
                    in_features
                    if i == 0
                    else min(max_features, block_expansion * (2**i))
                ),
                "out_features": min(max_features, block_expansion * (2 ** (i + 1))),
                "kernel_size": 3,
                "padding": 1,
            }
            for i in range(num_blocks)
        ]
        self.layers = eqx.nn.Sequential(
            [DownBlock2D(**params) for params in layers_params]
        )

    def __call__(self, x, state):

        outputs = [x]

        for layer in self.layers:
            x, state = layer(x, state)
            outputs.append(x)

        return outputs, state


class Decoder(eqx.Module):
    layers: eqx.nn.Sequential
    output_feature_count: int

    def __init__(
        self, key, block_expansion, input_features, num_blocks=3, max_features=256
    ):

        keys = jax.random.split(key, num_blocks)

        layers_params = [
            {
                "key": keys[i],
                # NOTE: explanation of magic number 2: skip connection addition doubles the number of features
                "in_features": (1 if i == (num_blocks - 1) else 2)
                * min(max_features, block_expansion * (2 ** (i + 1))),
                "out_features": min(max_features, block_expansion * (2**i)),
                "kernel_size": 3,
                "padding": 1,
            }
            for i in reversed(range(num_blocks))
        ]

        self.layers = eqx.nn.Sequential(
            [UpBlock2D(**params) for params in layers_params]
        )

        self.output_feature_count = (
            block_expansion + input_features
        )  # this is why we need input_features

    def __call__(self, inputs, state):

        inputs = inputs[::-1]

        x = inputs[0]

        for layer, skip in zip(self.layers, inputs[1:]):

            x, state = layer(x, state)

            x = jax.numpy.concat((x, skip))

        return x, state


class HourGlass(eqx.Module):
    encoder: eqx.Module
    decoder: eqx.Module
    head: eqx.nn.Conv2d

    def __init__(
        self,
        key,
        block_expansion,
        in_features,
        out_features,
        num_blocks=3,
        max_features=256,
    ):

        encoder_key, decoder_key, conv_key = jax.random.split(key, 3)
        self.encoder = Encoder(
            encoder_key, block_expansion, in_features, num_blocks, max_features
        )
        self.decoder = Decoder(
            decoder_key, block_expansion, in_features, num_blocks, max_features
        )

        self.head = eqx.nn.Conv2d(
            key=conv_key,
            kernel_size=1,
            in_channels=self.decoder.output_feature_count,
            out_channels=out_features,
            padding=0,
        )

    # state is a tuple
    def __call__(self, input_, state):

        x, state = self.encoder(input_, state)
        x, state = self.decoder(x, state)

        return self.head(x), state
