"""Run the locked 216-architecture replication on cropped SVHN.

This wrapper reuses the validated CIFAR-10 training engine so that model
construction, optimization, callbacks, seeds, and resume behavior remain
identical. It changes only the dataset loader and output namespace.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import train_cifar10_full216 as core


SOURCE_TRAIN_SIZE = 73257
SOURCE_TEST_SIZE = 26032
TRAIN_SUBSET_SIZE = 50000
TEST_SUBSET_SIZE = 10000
TRAIN_SUBSET_SEED = 20260929
TEST_SUBSET_SEED = 20260930


def proportional_quotas(labels, sample_size):
    classes, counts = np.unique(labels, return_counts=True)
    exact = counts.astype(np.float64) * sample_size / len(labels)
    quotas = np.floor(exact).astype(int)
    remainder = int(sample_size - quotas.sum())

    if remainder:
        fractional = exact - quotas
        order = np.lexsort((classes, -fractional))
        quotas[order[:remainder]] += 1

    if np.any(quotas > counts):
        raise RuntimeError("A stratified quota exceeds its class size.")
    return classes, quotas


def stratified_sample_indices(labels, sample_size, seed):
    labels = np.asarray(labels).reshape(-1)
    if not 1 <= sample_size <= len(labels):
        raise ValueError("sample_size must be between 1 and the dataset size.")

    classes, quotas = proportional_quotas(labels, sample_size)
    rng = np.random.default_rng(seed)
    selected = []

    for class_id, quota in zip(classes, quotas):
        class_indices = np.flatnonzero(labels == class_id)
        rng.shuffle(class_indices)
        selected.extend(class_indices[:quota])

    selected = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected)
    if len(selected) != sample_size:
        raise RuntimeError("The stratified sample has an unexpected size.")
    return selected


def stratified_split_exact(labels, fraction, seed):
    labels = np.asarray(labels).reshape(-1)
    validation_size = int(round(len(labels) * fraction))
    validation_indices = stratified_sample_indices(
        labels,
        validation_size,
        seed,
    )
    validation_mask = np.zeros(len(labels), dtype=bool)
    validation_mask[validation_indices] = True
    train_indices = np.flatnonzero(~validation_mask)

    rng = np.random.default_rng(seed)
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return train_indices, validation_indices


def load_svhn_mat(path):
    try:
        from scipy.io import loadmat
    except ImportError as error:
        raise RuntimeError(
            "SciPy is required to read the official SVHN .mat files."
        ) from error

    if not path.exists():
        raise FileNotFoundError(
            f"SVHN file not found: {path}. Download train_32x32.mat and "
            "test_32x32.mat before starting workers."
        )

    content = loadmat(path, variable_names=("X", "y"))
    images = np.asarray(content["X"], dtype=np.uint8).transpose(3, 0, 1, 2)
    labels = np.asarray(content["y"], dtype=np.int64).reshape(-1) % 10
    return images, labels


def load_svhn_fixed_subsets():
    x_source_train, y_source_train = load_svhn_mat(
        core.DATA_DIR / "train_32x32.mat"
    )
    x_source_test, y_source_test = load_svhn_mat(
        core.DATA_DIR / "test_32x32.mat"
    )

    if len(x_source_train) != SOURCE_TRAIN_SIZE:
        raise RuntimeError(
            f"Unexpected SVHN train size: {len(x_source_train)}"
        )
    if len(x_source_test) != SOURCE_TEST_SIZE:
        raise RuntimeError(
            f"Unexpected SVHN test size: {len(x_source_test)}"
        )

    train_subset = stratified_sample_indices(
        y_source_train,
        TRAIN_SUBSET_SIZE,
        TRAIN_SUBSET_SEED,
    )
    test_subset = stratified_sample_indices(
        y_source_test,
        TEST_SUBSET_SIZE,
        TEST_SUBSET_SEED,
    )

    x_all = x_source_train[train_subset]
    y_all = y_source_train[train_subset]
    x_test = x_source_test[test_subset]
    y_test = y_source_test[test_subset]

    expected_train_shape = (TRAIN_SUBSET_SIZE, 32, 32, 3)
    expected_test_shape = (TEST_SUBSET_SIZE, 32, 32, 3)
    if x_all.shape != expected_train_shape or x_test.shape != expected_test_shape:
        raise RuntimeError(
            f"Unexpected SVHN subset shapes: train={x_all.shape}, "
            f"test={x_test.shape}"
        )

    return (x_all, y_all), (x_test, y_test)


core.DATASET_NAME = "svhn"
core.DATASET_DISPLAY_NAME = "SVHN"
core.NUM_CLASSES = 10
core.DATASET_METADATA = {
    "source_train_samples": SOURCE_TRAIN_SIZE,
    "source_test_samples": SOURCE_TEST_SIZE,
    "selected_train_plus_validation_samples": TRAIN_SUBSET_SIZE,
    "selected_test_samples": TEST_SUBSET_SIZE,
    "train_subset_seed": TRAIN_SUBSET_SEED,
    "test_subset_seed": TEST_SUBSET_SEED,
    "uses_svhn_extra_split": False,
}
core.load_dataset = load_svhn_fixed_subsets
core.stratified_split = stratified_split_exact


if __name__ == "__main__":
    core.main()
