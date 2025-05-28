import os
from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import utils


def dump_to_folder(index, image, keypoints, actual_keypoints, folder):

    os.makedirs(folder, exist_ok=True)

    plt.imshow(image[0], cmap="gray")
    plt.scatter(keypoints[:, 0], keypoints[:, 1], c="red")
    plt.scatter(actual_keypoints[:, 0], actual_keypoints[:, 1], c="green")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(folder, f"eval_{index}.png"), bbox_inches="tight", pad_inches=0
    )
    plt.close()


# NOTE: impure function with side effects
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
        pred_keypoints = utils.batch_softargmax_heatmaps(pred_heatmaps)

        eval_keypoints = batch_y

        error = jnp.mean(jnp.linalg.norm(pred_keypoints - eval_keypoints, axis=-1))
        all_errors.append(error)
        dump_to_folder(
            counter, batch_x[3], pred_keypoints[3], eval_keypoints[3], "./eval"
        )
        counter += 1

    return float(jnp.mean(jnp.array(all_errors)))


def visualize_training(
    image,
    true_keypoints,
    heatmaps,
    alpha=0.5,
    cmap="jet",
    directory="train",
    filename="test",
):
    """
    image: [H, W] or [1, H, W] grayscale image (float32 or uint8)
    heatmaps: [K, H, W] predicted or true heatmaps
    keypoints: [K, 2] optional keypoint coordinates (x, y)
    """

    os.makedirs(directory, exist_ok=True)

    if image.ndim == 3:
        image = image[0]  # [1, H, W] → [H, W]

    H, W = image.shape

    fig, axs = plt.subplots(ncols=2, nrows=6, figsize=(5, 5), constrained_layout=True)

    heatmaps = utils.spatial_softmax(heatmaps, temp=10)
    true_heatmaps = utils.generate_heatmaps_from_keypoints(true_keypoints, (64, 64))

    for col in range(2):
        for row in range(6):
            ax = axs[row, col]
            ax.imshow(image, cmap="gray")

            if row == 0:
                continue

            hm = true_heatmaps if col == 0 else heatmaps

            ax.imshow(hm[row - 1], cmap=cmap, alpha=alpha)
            ax.axis("off")

    plt.savefig(os.path.join(directory, f"{filename}.png"))
    plt.close()
