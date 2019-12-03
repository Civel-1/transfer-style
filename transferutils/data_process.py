from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from PIL import Image
from tqdm import tqdm


def import_transform(img_path: str,
                     output_shape: Sequence[int] = (512, 512, 3),
                     mode: str = "tf",
                     interp: int = Image.ANTIALIAS) -> np.ndarray:

    # Import and resize
    if output_shape[0] == output_shape[1]:
        img = Image.open(img_path).resize(output_shape[0:2], interp)
    else:
        raise ValueError(f"Output shape for image must be a square ({output_shape[0]} != {output_shape[1]}).")

    # Simulate batching
    img = np.expand_dims(np.array(img), axis=0)

    # Apply ImageNet preprocess (Library dependent)
    if mode in ["tf", "caffe", "torch"]:
        img = keras.applications.vgg19.preprocess_input(img, mode=mode)
    else:
        raise ValueError(f"Mode must be one of 'tf', 'caffe' or 'torch'. (not {mode})")

    return img


def deprocess_img(img: np.ndarray, mode: str = "tf") -> np.ndarray:
    # TODO: support all other preprocessing modes
    # WARNING: only implemented for TensorFlow preprocessing

    # Deprocessing function inferred from keras_applications
    # source code.
    if mode == "tf":
        deprocessed = img.copy()
        deprocessed += 1.
        deprocessed *= 127.5
        return deprocessed.astype("uint8")
    else:
        raise NotImplementedError()