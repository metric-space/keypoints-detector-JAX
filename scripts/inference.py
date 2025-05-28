import os
import sys

import equinox as eqx
import jax
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import dataset
import model
import utils
from config import Config
from utils import load_model

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("usage: python script/inference.py <image>")
        sys.exit(1)

    img_path = sys.argv[1]

    config = Config()

    key = jax.random.PRNGKey(3456)

    model, state = eqx.nn.make_with_state(model.HourGlass)(
        key=key,
        block_expansion=config.block_expansion,
        in_features=config.input_channels,
        out_features=config.output_channels,
        num_blocks=config.num_blocks,
        max_features=config.max_features,
    )

    (model, state) = load_model(
        (model, state), os.path.join(config.models_directory, config.model_filename)
    )

    inference_model = eqx.nn.inference_mode(model)
    inference_model = eqx.Partial(inference_model, state=state)

    img_np = dataset.datum_preprocessing_pipeline(Image.open(img_path))

    output, state = inference_model(img_np)

    # should be made into a post processing pipeline
    keypoints = utils.softargmax_heatmaps(output)

    plt.imshow(img_np[0], cmap="gray")
    plt.scatter(keypoints[:, 0], keypoints[:, 1], c="red")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("./output.png", bbox_inches="tight", pad_inches=0)
    plt.close()
