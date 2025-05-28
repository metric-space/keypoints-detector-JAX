from utils import load_model
import model
import utils

import jax
import equinox as eqx

import sys

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from config import Config

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("usage: python script/inference.py <image>")
        sys.exit(1)

    img_path = sys.argv[1]

    config = Config()

    key = jax.random.PRNGKey(3456)

    model, state = eqx.nn.make_with_state(model.HourGlass)(
             key = key,
             block_expansion = config.block_expansion,
             in_features = config.input_channels,
             out_features = config.output_channels,
             num_blocks = config.num_blocks,
             max_features = config.max_features,
    )

    (model, state) = load_model((model, state), "./models/keypoints.eqx")

    inference_model = eqx.nn.inference_mode(model)
    inference_model = eqx.Partial(inference_model, state=state)


    # should be made into a pre processing pipeline
    img = Image.open(img_path).resize(config.image_size)
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_np = np.moveaxis(img_np, -1, 0)

    output, state = inference_model(img_np)

    # should be made into a post processing pipeline
    keypoints = utils.softargmax_heatmaps(output)

    plt.imshow(img_np[0], cmap="gray")
    plt.scatter(keypoints[:, 0], keypoints[:, 1], c="red")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig( "./output.png", bbox_inches="tight", pad_inches=0)
    plt.close()
