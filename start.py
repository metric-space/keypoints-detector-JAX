import os
import urllib.request
import zipfile
from PIL import Image
import matplotlib.pyplot as plt
import grain
import numpy as np
import jax
import equinox as eqx
import optax
from functools import partial


def _load_landmarks(landmark_file):
    with open(landmark_file, "r") as f:
        lines = f.readlines()[2:]  # skip headers
    entries = [line.strip().split() for line in lines]
    filenames = [e[0] for e in entries]
    landmarks = np.array([[int(x) for x in e[1:]] for e in entries])
    return filenames, landmarks


def generate_heatmaps_from_keypoints(keypoints, heatmap_size, sigma=2.0):
    B, K, _ = keypoints.shape
    H, W = heatmap_size
    y = jnp.arange(0, H)[:, None]
    x = jnp.arange(0, W)[None, :]
    y = jnp.tile(y, (1, W))
    x = jnp.tile(x, (H, 1))

    def gen_single(b, k):
        cx, cy = keypoints[b, k]
        return jnp.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))

    return [[gen_single(b, k) for k in range(K)] for b in range(B)]


def celeba_keypoints(directory, image_size=(96, 96), max_samples=None):

    landmark_file = os.path.join(directory,"list_landmarks_celeba.txt")
    image_folder = os.path.join(directory, "img_align_celeba")


    filenames, landmarks = _load_landmarks(landmark_file)

    images = []
    keypoints = []

    for idx, (filename, lm) in enumerate(zip(filenames, landmarks)):
        if max_samples and idx >= max_samples:
            break
        img_path = os.path.join(image_folder, filename)
        img = Image.open(img_path).convert("L") #.resize(image_size)
        img_np = np.array(img, dtype=np.float32) / 255.0
        images.append(img_np[None, ...])  # Add channel dim
        keypoints.append(lm.reshape(5, 2))  # (x1,y1,...,x5,y5) → [[x,y], ...]

    keypoints = generate_heatmaps_from_keypoints(kepoints, images[0][0].shape)

    train_length = int(len(images)*0.8)

    train_data = images[:train_length]
    train_keypoints = keypoints[:train_length]

    test_data = images[train_length:]
    test_keypoints = keypoints[train_length:]

    return np.stack(train_data), np.stack(train_keypoints), np.stack(test_data), np.stack(test_keypoints)


class UpBlock2D(eqx.Module):
    layers: eqx.nn.Sequential

    def __init__(self, key, in_features, out_features, kernel_size=3, padding=1, groups=1):
        self.layers = eqx.nn.Sequential([
            eqx.nn.Conv2d(key=key, kernel_size=kernel_size, in_channels=in_features, out_channels=out_features, padding=padding, groups=groups),
            eqx.nn.BatchNorm(input_size=out_features, axis_name="batch"),
            eqx.nn.Lambda(jax.nn.relu)
            ])

    def __call__(self, x, state):

        # I assume [C,W,H]
        C,W,H = x.shape

        scale = 2
        out = jax.image.resize(x,shape=(C, scale*W, scale*H), method="nearest")
        return self.layers(out, state)


class DownBlock2D(eqx.Module):
    layers: eqx.nn.Sequential

    def __init__(self, key, in_features, out_features, kernel_size=3, padding=1, groups=1):
        self.layers = eqx.nn.Sequential([
            eqx.nn.Conv2d(key=key, kernel_size=kernel_size, in_channels=in_features, out_channels=out_features, padding=padding, groups=groups),
            eqx.nn.BatchNorm(input_size=out_features, axis_name="batch"),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.AvgPool2d(kernel_size=2,stride=2) # check if the authors are overdoing it with (2,2)
            ])

    def __call__(self, x, state):

        return self.layers(x, state)


class Encoder(eqx.Module):
    layers: eqx.nn.Sequential

    def __init__(self, key, block_expansion, in_features, num_blocks=3, max_features=256):

        keys = jax.random.split(key,num_blocks)

        layers_params = [{"key": keys[i],
                        "in_features": in_features if i == 0 else min(max_features, block_expansion*(2**i)),
                        "out_features": min(max_features, block_expansion*(2**(i + 1))), 
                        "kernel_size": 3, 
                        "padding": 1
                        } for i in range(num_blocks)]
        self.layers = eqx.nn.Sequential([DownBlock2D(**params) for params in layers_params])

    def __call__(self, x, state):

        outputs = [x]

        # pretty sure there 
        for layer in self.layers:
            x, state = layer(x, state)
            outputs.append(x)

        print([x.shape for x in outputs])

        return outputs, state


class Decoder(eqx.Module):
    layers: eqx.nn.Sequential
    output_feature_count: int

    def __init__(self, key, block_expansion, num_blocks=3, max_features=256):

        keys = jax.random.split(key, num_blocks)

        layers_params = [{"key": keys[i],
                          # NOTE: explanation of magic number 2: skip connection addition doubles the number of features
                          "in_features": (1 if i == (num_blocks-1) else 2)*min(max_features, block_expansion*(2**(i+1))),
                          "out_features": min(max_features, block_expansion*(2**i)), 
                          "kernel_size": 3, 
                          "padding": 1} for i in reversed(range(num_blocks))]

        self.layers = eqx.nn.Sequential([UpBlock2D(**params) for params in layers_params])

        self.output_feature_count = layer_params[-1]['out_features']

    def __call__(self, inputs, state):

        inputs = inputs[::-1]

        x = inputs[0]

        print("Input shape is ", x.shape)

        for layer, skip in zip(self.layers, inputs[1:]):

            x, state = layer(x, state)

            print("output shape is ", x.shape)

            x = jax.numpy.concat((x, skip))

        return x, state


class HourGlass(eqx.Module):
    encoder: eqx.Module
    decoder: eqx.Module
    conv: eqx.nn.Conv2d

    def __init__(self, key, block_expansion, in_features, out_features, num_blocks=3, max_features=256):

        encoder_key, decoder_key, conv_key = jax.random.split(key, 3)
        self.encoder = Encoder(encoder_key, block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(decoder_key, block_expansion, num_blocks, max_features)

        self.conv = eqx.nn.conv(key=conv_key, kernel_size=1, in_channels=self.decoder.output_feature_count, out_channels=out_features, padding=0)

    # state is a tuple
    def __call__(self, input_, state):

        x, state = self.encoder(input_, state)
        x, state = self.decoder(x, state)

        return self.conv(x), state


@partial(eqx.filter_value_and_grad,has_aux=True)
def loss_fn(model, x, y, state):
    model = jax.vmap(model, axis_name="batch", in_axes=(0,None), out_axes=(0,None))
    pred, state = model(x, state)
    return jnp.mean((pred - y) ** 2), state


@eqx.filter_jit
def make_step(model, state, opt_state, x, y, optimizer):
    (loss, state), grads = loss_fn(model, x, y, state)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, state


if __name__ == '__main__':
    train_images, train_keypoints, test_images, test_keypoints = celeba_keypoints("./data/celeba")

    batch_size = 30

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    trainloader = grain.load(
        list(zip(train_images, train_keypoints)),
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    testloader = grain.load(
        list(zip(test_images, test_images)),
        batch_size=batch_size,
        num_epochs=1,
        shuffle=True,
        seed=seed,
    )

    model, state = eqx.nn.make_with_state(HourGlass)(key, 1, 5, num_blocks=2)

    model, opt_state, loss = make_step(model, opt_state, x, y, optimizer)

    for step, (batch_x, batch_y) in zip(range(steps), trainloader):

        model, opt_state, loss, state = make_step(model, state, opt_state, batch_x, batch_y, optimizer)

        print(loss)



    #index = 1
    #img = images[index][0]
    #kps = keypoints[index]
    #plt.imshow(img, cmap="gray")
    #plt.scatter(kps[:, 0], kps[:, 1], c="red")
    #plt.axis("off")
    #plt.tight_layout()
    #plt.savefig('./out.png', bbox_inches="tight", pad_inches=0)
    #plt.close()
