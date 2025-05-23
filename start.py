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
import gdown
import jax.numpy as jnp

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


def generate_heatmaps_from_keypoints(keypoints, heatmap_size, sigma=1.0):
    B, K = len(keypoints), keypoints[0].shape[0]
    H, W = heatmap_size
    y = jnp.arange(0, H)[:, None]
    x = jnp.arange(0, W)[None, :]
    y = jnp.tile(y, (1, W))
    x = jnp.tile(x, (H, 1))

    def gen_single(b, k):
        cx, cy = keypoints[b][k]
        return jnp.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))

    return [[gen_single(b, k) for k in range(K)] for b in range(B)]


def heatmap_to_keypoints(heatmaps):
    """Convert [B, K, H, W] heatmaps to [B, K, 2] (x, y) coords via argmax."""
    B, K, H, W = heatmaps.shape
    flat = heatmaps.reshape(B, K, -1)
    idx = jnp.argmax(flat, axis=-1)
    y, x = jnp.divmod(idx, W)
    return jnp.stack([x, y], axis=-1)  # [B, K, 2]


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

    #if not os.path.exists(directory):
    _download_celeba(directory)
    #else:
    #    print("Skipping download action as directory exists")

    landmark_file = os.path.join(directory,"list_landmarks_celeba.txt")
    image_folder = os.path.join(directory, "img_align_celeba")


    filenames, landmarks = _load_landmarks(landmark_file)

    images = []
    keypoints = []

    for idx, (filename, lm) in enumerate(zip(filenames, landmarks)):
        if max_samples and idx >= max_samples:
            break
        img_path = os.path.join(image_folder, filename)
        img = Image.open(img_path).convert("L").resize((64, 64))
        img_np = np.array(img, dtype=np.float32) / 255.0
        #img_np = np.moveaxis(img_np, -1, 0)
        images.append(img_np[None, ...])  # Add channel dim

        lm = resize_keypoints(lm.reshape(5,2), (218, 178), (64,64))
        keypoints.append(lm)  # (x1,y1,...,x5,y5) → [[x,y], ...]

    keypoints = generate_heatmaps_from_keypoints(keypoints, (64,64))

    print(f"Length of images is {len(images)}")

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

        return outputs, state


class Decoder(eqx.Module):
    layers: eqx.nn.Sequential
    output_feature_count: int

    def __init__(self, key, block_expansion, input_features, num_blocks=3, max_features=256):

        keys = jax.random.split(key, num_blocks)

        layers_params = [{"key": keys[i],
                          # NOTE: explanation of magic number 2: skip connection addition doubles the number of features
                          "in_features": (1 if i == (num_blocks-1) else 2)*min(max_features, block_expansion*(2**(i+1))),
                          "out_features": min(max_features, block_expansion*(2**i)), 
                          "kernel_size": 3, 
                          "padding": 1} for i in reversed(range(num_blocks))]

        self.layers = eqx.nn.Sequential([UpBlock2D(**params) for params in layers_params])

        self.output_feature_count = block_expansion + input_features # this is why we need input_features

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

    def __init__(self, key, block_expansion, in_features, out_features, num_blocks=3, max_features=256):

        encoder_key, decoder_key, conv_key = jax.random.split(key, 3)
        self.encoder = Encoder(encoder_key, block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(decoder_key, block_expansion, in_features, num_blocks, max_features)

        self.conv = eqx.nn.Conv2d(key=conv_key, kernel_size=1, in_channels=self.decoder.output_feature_count, out_channels=out_features, padding=0)

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

def compute_eval_metrics(pred_keypoints, true_keypoints):
    dists = jnp.linalg.norm(pred_keypoints - true_keypoints, axis=-1)  # [B, K]
    return jnp.mean(dists)  # scalar


def soft_argmax(heatmaps, beta=100.0):

    B, K, H, W = heatmaps.shape
    flat = heatmaps.reshape(B, K, -1)

    softmaxed = jax.nn.softmax(flat * beta, axis=-1)  # [B, K, H*W]

    coords = jnp.arange(H * W)
    expected = jnp.sum(softmaxed * coords, axis=-1)  # [B, K]
    y, x = jnp.divmod(expected, W)
    return jnp.stack([x, y], axis=-1)


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

    inference_model = jax.vmap(inference_model, axis_name="batch") #, in_axes=(0,None), out_axes=(0,None))

    for batch_x, batch_y in test_loader:

        pred_heatmaps, _ = inference_model(batch_x)
        pred_keypoints = soft_argmax(pred_heatmaps)
        eval_keypoints = heatmap_to_keypoints(batch_y)

        error = compute_eval_metrics(pred_keypoints, eval_keypoints)
        all_errors.append(error)

    return float(jnp.mean(jnp.array(all_errors)))


if __name__ == '__main__':
    train_images, train_keypoints, test_images, test_keypoints = celeba_keypoints("./data", max_samples=12000)

    batch_size = 30

    optimizer = optax.adam(1e-3)

    seed = 3728

    key = jax.random.PRNGKey(1023)

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

    model, state = eqx.nn.make_with_state(HourGlass)(key, 32, 1, 5, num_blocks=5, max_features=512)

    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    for step, (batch_x, batch_y) in zip(range(200), trainloader):

        model, opt_state, loss, state = make_step(model, state, opt_state, batch_x, batch_y, optimizer)

        print(loss)

        if step % 10 == 0:

            print("Eval :", evaluate_model(model, state, testloader))


    #index = 4
    #img = train_images[index][0]
    #kps = train_keypoints[index]
    #plt.imshow(img) #cmap="c")
    #plt.scatter(kps[:, 0], kps[:, 1], c="red")
    #plt.axis("off")
    #plt.tight_layout()
    #plt.savefig('./out.png', bbox_inches="tight", pad_inches=0)
    #plt.close()
