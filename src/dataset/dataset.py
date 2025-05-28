import os
import sys
import urllib.request
import zipfile

import gdown
import grain
import numpy as np
from PIL import Image

import utils


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


def datum_preprocessing_pipeline(image: Image, size=(64, 64)) -> Image:
    img = image.resize((64, 64))
    img_np = np.array(img, dtype=np.float32) / 255.0
    return np.moveaxis(img_np, -1, 0)


def celeba_keypoints(directory, resized_dims, max_samples=None):

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
        img = Image.open(img_path)
        images.append(datum_preprocessing_pipeline(img, resized_dims))

        lm = utils.resize_keypoints(lm.reshape(5, 2), (218, 178), resized_dims)
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


def celeba_train_test_dataloaders(
    directory, resized_dims, max_samples=None, batch_size=30, data_seed=1234
):
    train_images, train_keypoints, test_images, test_keypoints = celeba_keypoints(
        directory, resized_dims=resized_dims, max_samples=max_samples
    )

    trainloader = grain.load(
        list(zip(train_images, train_keypoints)),
        batch_size=batch_size,
        shuffle=True,
        seed=data_seed,
    )

    testloader = grain.load(
        list(zip(test_images, test_keypoints)),
        batch_size=batch_size,
        num_epochs=1,
        shuffle=True,
        seed=data_seed,
    )

    return trainloader, testloader
