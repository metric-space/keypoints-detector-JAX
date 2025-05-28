from utils import load_model
import model
import utils

import jax
import equinox as eqx

import sys

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

if __name__ == '__main__':

    key = jax.random.PRNGKey(3456)

    model, state = eqx.nn.make_with_state(model.HourGlass)(
        key, 32, 3, 5, num_blocks=5, max_features=256
    )

    (model, state) = load_model((model, state), "./models/keypoints.eqx")

    inference_model = eqx.nn.inference_mode(model)
    inference_model = eqx.Partial(inference_model, state=state)

    img_path = sys.argv[1]

    img = Image.open(img_path).resize((64, 64))
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_np = np.moveaxis(img_np, -1, 0)

    output, state = inference_model(img_np)

    keypoints = utils.softargmax_heatmaps(output)

    plt.imshow(img_np[0], cmap="gray")
    plt.scatter(keypoints[:, 0], keypoints[:, 1], c="red")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig( "./output.png", bbox_inches="tight", pad_inches=0)
    plt.close()
