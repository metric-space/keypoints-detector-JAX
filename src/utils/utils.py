import os
from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


def save_model(
    model: eqx.Module, state: eqx.nn.State, filename: str, directory: str = "./models"
):

    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, filename), "wb") as f:
        eqx.tree_serialise_leaves(f, (model, state))


def load_model(model: eqx.Module, filename: str) -> Tuple[eqx.Module, eqx.nn.State]:
    with open(filename, "rb") as f:
        return eqx.tree_deserialise_leaves(f, model)


# no batch expected here
def generate_heatmaps_from_keypoints(keypoints, heatmap_size, sigma=1):
    K = keypoints.shape[0]  # [(x_0, y_0), (x_1, y_1), (y_2, y_2), ...]
    H, W = heatmap_size
    y = jnp.arange(0, H)[:, None]
    x = jnp.arange(0, W)[None, :]

    def gen_single(k):
        cx, cy = keypoints[k]
        return jnp.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))

    return [gen_single(k) for k in range(K)]


def spatial_softmax(heatmaps, temp):

    K, H, W = heatmaps.shape
    flat = heatmaps.reshape(K, -1)
    norm = jax.nn.softmax(flat * temp, axis=-1)
    return norm.reshape(K, H, W)


def batch_spatial_softmax(heatmaps):
    """
    Apply softmax over spatial dimensions of [B, K, H, W] tensor.
    Returns: normalized heatmaps of same shape
    """
    return jax.vmap(spatial_softmax)(heatmaps)


def softargmax_heatmaps(heatmaps):
    """
    Convert [K, H, W] heatmaps into [K, 2] keypoints using softargmax.
    Args:
        heatmaps: predicted heatmaps [K, H, W]
        normalize: if True, outputs coords in [-1, 1], else in pixel units
    Returns:
        coords: [K, 2] — soft (x, y) keypoints
    """
    K, H, W = heatmaps.shape

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
    grid = grid[None, :, :, :]  # [ 1, H, W, 2]

    # Apply softmax over spatial dims
    weights = spatial_softmax(heatmaps, temp)[..., None]  # [K, H, W, 1]

    # Compute expected (x, y)
    coords = jnp.sum(weights * grid, axis=(1, 2))  # [K, 2]
    return coords


def batch_softargmax_heatmaps(heatmaps):
    """
    Apply softmax over spatial dimensions of [B, K, H, W] tensor.
    Returns: normalized heatmaps of same shape
    """
    return jax.vmap(softargmax_heatmaps)(heatmaps)


def resize_keypoints(keypoints, original_size, new_size):
    """
    keypoints: [K, 2] array (x, y) coordinates
    original_size: (H_orig, W_orig)
    new_size: (H_new, W_new)
    """
    scale_x = new_size[1] / original_size[1]
    scale_y = new_size[0] / original_size[0]
    return keypoints * np.array([scale_x, scale_y])
