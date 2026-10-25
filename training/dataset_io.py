"""Read the official extracted CIFAR-100 Python archive without cache changes."""
from pathlib import Path
import pickle

import numpy as np


def load_cifar100(data_dir):
    root = Path(data_dir) / "cifar-100-python"

    def read_batch(name, expected):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Extract the official CIFAR-100 Python archive first: {path}")
        # Pickle must only be loaded from the trusted official dataset archive.
        with path.open("rb") as stream:
            batch = pickle.load(stream, encoding="bytes")
        images = np.asarray(batch[b"data"], dtype=np.uint8)
        images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = np.asarray(batch[b"fine_labels"], dtype=np.int64)
        if images.shape != (expected, 32, 32, 3) or labels.shape != (expected,):
            raise ValueError(f"Unexpected CIFAR-100 dimensions in {path}")
        return images, labels

    return read_batch("train", 50000), read_batch("test", 10000)
