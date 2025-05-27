import os
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from functools import partial
from typing import Any

import equinox as eqx
import gdown
import grain
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax.tree_util import tree_flatten_with_path
from PIL import Image


def named_grad_norms(grads):
    flat = tree_flatten_with_path(grads)[0]
    return {
        ".".join(str(k) for k in path): jnp.sqrt(jnp.sum(leaf**2))
        for path, leaf in flat
        if leaf is not None
    }


def _download_celeba(dest_dir):
    os.makedirs(dest_dir, exist_ok=True)

    # === Download images ===
    img_id = "0B7EVK8r0v71pZjFTYXZWM3FlRnM"  # Official CelebA image zip on Google Drive
    img_zip = os.path.join(dest_dir, "img_align_celeba.zip")
    if not os.path.exists(img_zip):
        print("Downloading CelebA image zip...")
        gdown.download(id=img_id, output=img_zip, quiet=False)

    with zipfile.ZipFile(img_zip, "r") as zf:
        zf.extractall(dest_dir)

    # === Download landmarks ===
    landmark_id = "0B7EVK8r0v71pd0FJY3Blby1HUTQ"  # Official landmark file
    landmark_path = os.path.join(dest_dir, "list_landmarks_celeba.txt")
    if not os.path.exists(landmark_path):
        print("Downloading CelebA landmark file...")
        gdown.download(id=landmark_id, output=landmark_path, quiet=False)


def _load_landmarks(landmark_file):
    with open(landmark_file, "r") as f:
        lines = f.readlines()[2:]  # skip headers
    entries = [line.strip().split() for line in lines]
    filenames = [e[0] for e in entries]
    landmarks = np.array([[int(x) for x in e[1:]] for e in entries])
    return filenames, landmarks


def generate_heatmaps_from_keypoints(keypoints, heatmap_size, sigma=1):
    K = keypoints.shape[0]  # [(x_0, y_0), (x_1, y_1), (y_2, y_2), ...]
    H, W = heatmap_size
    y = jnp.arange(0, H)[:, None]
    x = jnp.arange(0, W)[None, :]

    def gen_single(k):
        cx, cy = keypoints[k]
        return jnp.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))

    return [gen_single(k) for k in range(K)]


def softargmax_heatmaps(heatmaps):
    """
    Convert [B, K, H, W] heatmaps into [B, K, 2] keypoints using softargmax.
    Args:
        heatmaps: predicted heatmaps [B, K, H, W]
        normalize: if True, outputs coords in [-1, 1], else in pixel units
    Returns:
        coords: [B, K, 2] — soft (x, y) keypoints
    """
    B, K, H, W = heatmaps.shape

    normalize = False
    temp = 10

    # Create grid
    if normalize:
        x_range = jnp.linspace(-1.0, 1.0, W)
        y_range = jnp.linspace(-1.0, 1.0, H)
    else:
        x_range = jnp.arange(W)
        y_range = jnp.arange(H)

    grid_x, grid_y = jnp.meshgrid(x_range, y_range)
    grid = jnp.stack([grid_x, grid_y], axis=-1)  # [H, W, 2]
    grid = grid[None, None, :, :, :]  # [1, 1, H, W, 2]

    # Apply softmax over spatial dims
    flat = heatmaps.reshape(B, K, -1)
    weights = jax.nn.softmax(flat * temp, axis=-1).reshape(B, K, H, W)[
        ..., None
    ]  # [B, K, H, W, 1]

    # Compute expected (x, y)
    coords = jnp.sum(weights * grid, axis=(2, 3))  # [B, K, 2]
    return coords


def resize_keypoints(keypoints, original_size, new_size):
    """
    keypoints: [K, 2] array (x, y) coordinates
    original_size: (H_orig, W_orig)
    new_size: (H_new, W_new)
    """
    scale_x = new_size[1] / original_size[1]
    scale_y = new_size[0] / original_size[0]
    return keypoints * np.array([scale_x, scale_y])


def celeba_keypoints(directory, max_samples=None):

    if not os.path.exists(directory):
        _download_celeba(directory)
    else:
        print("Skipping download action as directory exists")

    landmark_file = os.path.join(directory, "list_landmarks_celeba.txt")
    image_folder = os.path.join(directory, "img_align_celeba")

    filenames, landmarks = _load_landmarks(landmark_file)

    images = []
    keypoints = []

    for idx, (filename, lm) in enumerate(zip(filenames, landmarks)):
        if max_samples and idx >= max_samples:
            break
        img_path = os.path.join(image_folder, filename)
        img = Image.open(img_path).resize((64, 64))
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_np = np.moveaxis(img_np, -1, 0)

        images.append(img_np)

        lm = resize_keypoints(lm.reshape(5, 2), (218, 178), (64, 64))
        keypoints.append(lm)  # (x1,y1,...,x5,y5) → [[x,y], ...]

    train_length = int(len(images) * 0.8)

    train_data = images[:train_length]
    train_keypoints = keypoints[:train_length]

    test_data = images[train_length:]
    test_keypoints = keypoints[train_length:]

    return (
        np.stack(train_data),
        np.stack(train_keypoints),
        np.stack(test_data),
        np.stack(test_keypoints),
    )


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
    conv: eqx.nn.Conv2d

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

        self.conv = eqx.nn.Conv2d(
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

        return self.conv(x), state


def spatial_softmax(heatmaps):

    K, H, W = heatmaps.shape
    flat = heatmaps.reshape(K, -1)
    norm = jax.nn.log_softmax(flat, axis=-1)
    return norm.reshape(K, H, W)


def batch_spatial_softmax(heatmaps):
    """
    Apply softmax over spatial dimensions of [B, K, H, W] tensor.
    Returns: normalized heatmaps of same shape
    """
    return jax.vmap(spatial_softmax)(heatmaps)


@partial(eqx.filter_value_and_grad, has_aux=True)
def loss_fn(model, x, y, state):
    model = jax.vmap(model, axis_name="batch", in_axes=(0, None), out_axes=(0, None))
    pred, state = model(x, state)
    loss1 = jnp.mean((softargmax_heatmaps(pred) - y) ** 2)
    return loss1, (state, pred)


@eqx.filter_jit
def make_step(model, state, opt_state, x, y, optimizer):
    (loss, aux), grads = loss_fn(model, x, y, state)
    state, pred = aux
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, state, pred, grads


def compute_eval_metrics(pred_keypoints, true_keypoints):
    """Returns average per-point L2 distance (pixel error)."""
    dists = jnp.linalg.norm(pred_keypoints - true_keypoints, axis=-1)  # [B, K]
    return jnp.mean(dists)  # scalar


def dump_to_folder(index, image, keypoints, actual_keypoints, folder):
    plt.imshow(image[0], cmap="gray")
    plt.scatter(keypoints[:, 0], keypoints[:, 1], c="red")
    plt.scatter(actual_keypoints[:, 0], actual_keypoints[:, 1], c="green")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(folder, f"eval_{index}.png"), bbox_inches="tight", pad_inches=0
    )
    plt.close()


def evaluate_model(model, state, test_loader):
    """
    Run the model on the evaluation dataset and compute average keypoint error.

    Args:
        model: Trained model
        eval_images: [N, C, H, W]
        eval_keypoints: [N, K, 2]
    Returns:
        mean_l2_error: float
    """
    all_errors = []

    inference_model = eqx.nn.inference_mode(model)
    inference_model = eqx.Partial(inference_model, state=state)

    inference_model = jax.vmap(
        inference_model, axis_name="batch"
    )  # , in_axes=(0,None), out_axes=(0,None))

    counter = 0

    for batch_x, batch_y in test_loader:

        pred_heatmaps, _ = inference_model(batch_x)
        pred_keypoints = softargmax_heatmaps(pred_heatmaps)

        eval_keypoints = batch_y

        error = compute_eval_metrics(pred_keypoints, eval_keypoints)
        all_errors.append(error)
        dump_to_folder(
            counter, batch_x[3], pred_keypoints[3], eval_keypoints[3], "./eval"
        )
        counter += 1

    return float(jnp.mean(jnp.array(all_errors)))


def visualize_training(
    image, true_keypoints, heatmaps, alpha=0.5, cmap="jet", filename="test"
):
    """
    image: [H, W] or [1, H, W] grayscale image (float32 or uint8)
    heatmaps: [K, H, W] predicted or true heatmaps
    keypoints: [K, 2] optional keypoint coordinates (x, y)
    """

    if image.ndim == 3:
        image = image[0]  # [1, H, W] → [H, W]

    H, W = image.shape

    fig, axs = plt.subplots(ncols=2, nrows=6, figsize=(5, 5), constrained_layout=True)

    heatmaps = spatial_softmax(heatmaps)
    true_heatmaps = generate_heatmaps_from_keypoints(true_keypoints, (64, 64))

    for col in range(2):
        for row in range(6):
            ax = axs[row, col]
            ax.imshow(image, cmap="gray")

            if row == 0:
                continue

            hm = true_heatmaps if col == 0 else heatmaps

            ax.imshow(hm[row - 1], cmap=cmap, alpha=alpha)
            ax.axis("off")

    plt.savefig(f"./{filename}.png")
    plt.close()


@dataclass
class Config:
    max_samples: int = 18000
    dataset_directory: str = "./data"

    batch_size: int = 30
    image_size: tuple[int, int] = (64, 64)

    lr: float = 4e-4
    lr_decay_steps: int = 100
    lr_alpha: float = 0.2

    input_channels: int = 3
    output_channels: int = 5
    max_features: int = 256
    num_blocks: int = 5
    block_expansion: int = 32

    data_seed: int = 3728
    nn_seed: int = 1023

    steps: int = 100


if __name__ == "__main__":

    config = Config()

    train_images, train_keypoints, test_images, test_keypoints = celeba_keypoints(
        "./data", max_samples=18000
    )

    batch_size = config.batch_size

    lr_schedule = optax.cosine_decay_schedule(
        init_value=config.lr, decay_steps=config.lr_decay_steps, alpha=config.lr_alpha
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(),
        optax.scale_by_schedule(lr_schedule),
        optax.scale(-1.0),
    )

    key = jax.random.PRNGKey(config.nn_seed)

    trainloader = grain.load(
        list(zip(train_images, train_keypoints)),
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.data_seed,
    )

    testloader = grain.load(
        list(zip(test_images, test_keypoints)),
        batch_size=config.batch_size,
        num_epochs=1,
        shuffle=True,
        seed=config.data_seed,
    )

    model, state = eqx.nn.make_with_state(HourGlass)(
        key, 32, 3, 5, num_blocks=5, max_features=256
    )

    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    for step, (batch_x, batch_y) in zip(range(100), trainloader):

        model, opt_state, loss, state, pred, grads = make_step(
            model, state, opt_state, batch_x, batch_y, optimizer
        )

        # print(named_grad_norms(grads))

        print(loss)

        i = 12

        if step % 10 == 0:
            visualize_training(
                batch_x[i], batch_y[i], pred[i], filename=f"./train/train_{step}_{i}"
            )

            print("Eval :", evaluate_model(model, state, testloader))
