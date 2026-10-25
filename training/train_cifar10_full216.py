"""Prepare and run the distributed 216-architecture CIFAR-10 experiment.

The architecture space, graph metrics, seeds, and data split protocol match the
CIFAR-100 experiment. CIFAR-10 outputs are always written to separate files;
CIFAR-100 training rows are never reused.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import pickle
import random
import time

# No global Keras cache is modified by this packaged version.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np
import pandas as pd


SEEDS = (11, 22, 33, 44, 55)
SPLIT_SEED = 123
VALIDATION_FRACTION = 0.10
BATCH_SIZE = 64
MAX_EPOCHS = 100
GROUPS = {"std": 1, "group2": 2, "group4": 4}
SELECTION_METRICS = (
    "avg_degree",
    "degree_entropy",
    "assortativity_coefficient",
)
DATASET_NAME = "cifar10"
DATASET_DISPLAY_NAME = "CIFAR-10"
NUM_CLASSES = 10
DATASET_METADATA = {}
DATA_DIR = Path(__file__).resolve().parent / "datasets"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--pilot-size", type=int, default=216)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--max-new-runs",
        type=int,
        help="Optional limit used for a smoke test.",
    )
    parser.add_argument(
        "--study-dir",
        default=str(Path(__file__).resolve().parent / "runs"),
    )
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--metrics-file", default=str(Path(__file__).resolve().parent / "graph_metrics_same_216.csv"))
    return parser.parse_args()


def atomic_write_csv(dataframe, path):
    temporary = path.with_name(path.name + ".tmp")
    dataframe.to_csv(temporary, index=False)
    os.replace(temporary, path)


def experiment_label(architecture_count):
    if architecture_count == 216:
        return "full216"
    return f"pilot{architecture_count}"


def select_maximin(metrics, count):
    values = metrics.loc[:, SELECTION_METRICS].to_numpy(dtype=np.float64)
    means = values.mean(axis=0)
    scales = values.std(axis=0, ddof=0)
    if np.any(scales == 0):
        raise RuntimeError("At least one pilot-selection metric is constant.")

    standardized = (values - means) / scales
    distance_to_center = np.sum(standardized**2, axis=1)
    selected = [int(np.argmax(distance_to_center))]
    minimum_distance = np.sum(
        (standardized - standardized[selected[0]]) ** 2,
        axis=1,
    )

    while len(selected) < count:
        minimum_distance[selected] = -1.0
        next_index = int(np.argmax(minimum_distance))
        selected.append(next_index)
        new_distance = np.sum(
            (standardized - standardized[next_index]) ** 2,
            axis=1,
        )
        minimum_distance = np.minimum(minimum_distance, new_distance)

    return selected


def assign_balanced_workers(selected, workers):
    quotient, remainder = divmod(len(selected), workers)
    capacities = [quotient + (worker < remainder) for worker in range(workers)]
    counts = [0] * workers
    loads = [0.0] * workers
    assignments = {}

    ordered = selected.sort_values(
        ["num_parameters_backbone", "architecture_index"],
        ascending=[False, True],
    )
    for row in ordered.itertuples(index=False):
        eligible = [
            worker
            for worker in range(workers)
            if counts[worker] < capacities[worker]
        ]
        worker = min(eligible, key=lambda item: (loads[item], counts[item], item))
        assignments[str(row.architecture_id)] = worker + 1
        counts[worker] += 1
        loads[worker] += float(row.num_parameters_backbone)

    result = selected.copy()
    result["worker_id"] = result["architecture_id"].map(assignments).astype(int)
    result = result.sort_values(["worker_id", "architecture_index"])
    result["worker_sequence"] = result.groupby("worker_id").cumcount() + 1
    return result


def prepare_assignments(study_dir, architecture_count, workers, metrics_path):
    metrics_path = Path(metrics_path)
    label = experiment_label(architecture_count)
    assignments_path = study_dir / f"{DATASET_NAME}_{label}_assignments.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics table not found: {metrics_path}")

    metrics = pd.read_csv(metrics_path).sort_values("architecture_index")
    if len(metrics) != 216 or metrics["architecture_id"].nunique() != 216:
        raise RuntimeError("The metrics table must contain 216 unique architectures.")
    if not 1 <= architecture_count <= len(metrics):
        raise ValueError("pilot-size must be between 1 and 216.")

    required = (*SELECTION_METRICS, "num_parameters_backbone")
    missing = [column for column in required if column not in metrics]
    if missing:
        raise RuntimeError(f"Missing required metrics columns: {missing}")

    if architecture_count == len(metrics):
        selected = metrics.copy()
        selected["pilot_selection_order"] = np.arange(1, len(metrics) + 1)
    else:
        selected_indices = select_maximin(metrics, architecture_count)
        selected = metrics.iloc[selected_indices].copy()
        selected["pilot_selection_order"] = np.arange(1, architecture_count + 1)

    assignments = assign_balanced_workers(selected, workers)
    atomic_write_csv(assignments, assignments_path)

    print(f"{DATASET_DISPLAY_NAME} assignments saved: {assignments_path}")
    print("Architectures per worker:")
    print(assignments.groupby("worker_id").size().to_string())
    print("Backbone-parameter load per worker:")
    print(
        assignments.groupby("worker_id")["num_parameters_backbone"]
        .sum()
        .round()
        .astype(int)
        .to_string()
    )
    return assignments_path


def stratified_split(labels, fraction, seed):
    labels = np.asarray(labels).reshape(-1)
    rng = np.random.default_rng(seed)
    train_indices = []
    validation_indices = []

    for class_id in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_id)
        rng.shuffle(class_indices)
        validation_count = int(round(len(class_indices) * fraction))
        validation_indices.extend(class_indices[:validation_count])
        train_indices.extend(class_indices[validation_count:])

    train_indices = np.asarray(train_indices, dtype=np.int64)
    validation_indices = np.asarray(validation_indices, dtype=np.int64)
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return train_indices, validation_indices


def load_cifar10_from_shared_cache():
    data_dir = DATA_DIR / "cifar-10-batches-py"
    train_paths = [data_dir / f"data_batch_{index}" for index in range(1, 6)]
    test_path = data_dir / "test_batch"
    missing = [path for path in (*train_paths, test_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Extracted CIFAR-10 cache is incomplete. Missing: "
            + ", ".join(str(path) for path in missing)
        )

    def load_batch(path):
        with path.open("rb") as handle:
            batch = pickle.load(handle, encoding="bytes")
        images = np.asarray(batch[b"data"], dtype=np.uint8)
        images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = np.asarray(batch[b"labels"], dtype=np.int64)
        return images, labels

    train_batches = [load_batch(path) for path in train_paths]
    x_all = np.concatenate([batch[0] for batch in train_batches], axis=0)
    y_all = np.concatenate([batch[1] for batch in train_batches], axis=0)
    x_test, y_test = load_batch(test_path)

    if x_all.shape != (50000, 32, 32, 3) or x_test.shape != (10000, 32, 32, 3):
        raise RuntimeError(
            f"Unexpected CIFAR-10 shapes: train={x_all.shape}, test={x_test.shape}"
        )
    return (x_all, y_all), (x_test, y_test)


def load_dataset():
    return load_cifar10_from_shared_cache()


def set_training_seed(tf, seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def conv_block(tf, x, filters, kernel_size, conv_type, stride, name):
    return tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=int(kernel_size),
        strides=int(stride),
        padding="same",
        groups=GROUPS[conv_type],
        activation="relu",
        use_bias=True,
        name=name,
    )(x)


def build_classifier(tf, config):
    inputs = tf.keras.Input(shape=(32, 32, 3), name="input")
    x = conv_block(tf, inputs, 16, 3, "std", 1, "conv0")
    x = conv_block(tf, x, 32, config.kernel1, config.conv1, 1, "conv1")
    x = conv_block(tf, x, 64, config.kernel2, config.conv2, 2, "conv2")
    x = conv_block(tf, x, 128, config.kernel3, config.conv3, 1, "conv3")
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="classifier",
    )(x)
    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=str(config.architecture_id).lower(),
    )


def make_summary(runs, assignments):
    if runs.empty:
        return pd.DataFrame()

    summary = (
        runs.groupby("architecture_id", as_index=False)
        .agg(
            completed_seeds=("seed", "nunique"),
            mean_test_acc=("test_acc", "mean"),
            std_test_acc=("test_acc", "std"),
            mean_test_loss=("test_loss", "mean"),
            mean_best_epoch=("best_epoch", "mean"),
            mean_training_time_s=("training_time_s", "mean"),
            num_parameters_total=("num_parameters_total", "first"),
        )
    )
    identity_candidates = [
        "architecture_id",
        "architecture_index",
        "model_name",
        "worker_id",
        "conv1",
        "conv2",
        "conv3",
        "kernel1",
        "kernel2",
        "kernel3",
        "num_parameters_backbone",
        "avg_degree",
        "degree_entropy",
        "normalized_degree_entropy",
        "assortativity_coefficient",
        "modularity",
    ]
    identity_columns = [
        column for column in identity_candidates if column in assignments.columns
    ]
    identity = assignments.loc[:, identity_columns].drop_duplicates("architecture_id")
    return identity.merge(summary, on="architecture_id", how="left")


def validate_existing_runs(runs, worker_architectures, worker_id):
    if runs.empty:
        return runs

    required = {
        "worker_id",
        "architecture_index",
        "architecture_id",
        "seed",
        "test_acc",
    }
    missing = required.difference(runs.columns)
    if missing:
        raise RuntimeError(f"Existing worker file is missing columns: {sorted(missing)}")

    allowed_architectures = set(worker_architectures["architecture_id"].astype(str))
    runs["architecture_id"] = runs["architecture_id"].astype(str)
    runs["seed"] = runs["seed"].astype(int)
    invalid = runs.loc[
        (runs["worker_id"].astype(int) != worker_id)
        | ~runs["architecture_id"].isin(allowed_architectures)
        | ~runs["seed"].isin(SEEDS)
    ]
    if not invalid.empty:
        raise RuntimeError(
            f"Existing {DATASET_DISPLAY_NAME} worker file contains rows from "
            "another assignment."
        )

    return runs.drop_duplicates(["architecture_id", "seed"], keep="last")


def train_worker(args, study_dir):
    if args.worker_id is None:
        raise ValueError("--worker-id is required unless --prepare-only is used.")
    if not 1 <= args.worker_id <= args.workers:
        raise ValueError("worker-id must be between 1 and workers.")
    if args.max_new_runs is not None and args.max_new_runs <= 0:
        raise ValueError("max-new-runs must be a positive integer.")

    label = experiment_label(args.pilot_size)
    assignments_path = study_dir / f"{DATASET_NAME}_{label}_assignments.csv"
    if not assignments_path.exists():
        raise FileNotFoundError(
            f"{DATASET_DISPLAY_NAME} assignments not found: {assignments_path}. "
            "Run --prepare-only first."
        )

    assignments = pd.read_csv(assignments_path)
    worker_architectures = assignments.loc[
        assignments["worker_id"] == args.worker_id
    ].sort_values("worker_sequence")
    if worker_architectures.empty:
        raise RuntimeError(f"No architectures assigned to worker {args.worker_id}.")

    import tensorflow as tf

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    (x_all, y_all), (x_test, y_test) = load_dataset()
    y_all = y_all.reshape(-1)
    y_test = y_test.reshape(-1)
    train_indices, validation_indices = stratified_split(
        y_all,
        VALIDATION_FRACTION,
        SPLIT_SEED,
    )

    x_train = x_all[train_indices].astype(np.float32) / 255.0
    y_train = y_all[train_indices]
    x_validation = x_all[validation_indices].astype(np.float32) / 255.0
    y_validation = y_all[validation_indices]
    x_test = x_test.astype(np.float32) / 255.0

    output_dir = study_dir / f"training_{DATASET_NAME}_{label}"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = output_dir / f"worker_{args.worker_id:02d}_runs.csv"
    summary_path = output_dir / f"worker_{args.worker_id:02d}_summary.csv"

    if runs_path.exists():
        runs = validate_existing_runs(
            pd.read_csv(runs_path),
            worker_architectures,
            args.worker_id,
        )
    else:
        runs = pd.DataFrame()

    completed = (
        set(zip(runs["architecture_id"].astype(str), runs["seed"].astype(int)))
        if not runs.empty
        else set()
    )

    metadata = {
        "dataset": DATASET_NAME,
        "num_classes": NUM_CLASSES,
        "worker_id": args.worker_id,
        "workers": args.workers,
        "seeds": list(SEEDS),
        "split_seed": SPLIT_SEED,
        "validation_fraction": VALIDATION_FRACTION,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "data_dir": str(DATA_DIR),
        "gpu_devices": [device.name for device in tf.config.list_physical_devices("GPU")],
        "train_samples": int(len(train_indices)),
        "validation_samples": int(len(validation_indices)),
        "test_samples": int(len(x_test)),
        **DATASET_METADATA,
    }
    (output_dir / f"worker_{args.worker_id:02d}_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(
        f"Reused {len(completed)} completed {DATASET_DISPLAY_NAME} run(s)."
    )
    print(
        f"Worker {args.worker_id}: {len(worker_architectures)} architectures, "
        f"{len(worker_architectures) * len(SEEDS)} total seed runs."
    )
    print(
        f"{DATASET_DISPLAY_NAME} data: train={len(train_indices)}, "
        f"validation={len(validation_indices)}, test={len(x_test)}"
    )

    new_runs_completed = 0
    for config in worker_architectures.itertuples(index=False):
        for seed in SEEDS:
            key = (str(config.architecture_id), int(seed))
            if key in completed:
                continue

            print(
                f"Starting {DATASET_DISPLAY_NAME} worker={args.worker_id}, "
                f"architecture={config.architecture_id}, seed={seed}"
            )
            tf.keras.backend.clear_session()
            gc.collect()
            set_training_seed(tf, seed)

            model = build_classifier(tf, config)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=["accuracy"],
            )
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=8,
                    min_delta=1e-4,
                    restore_best_weights=True,
                    verbose=0,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=4,
                    min_lr=1e-6,
                    verbose=0,
                ),
            ]

            start_time = time.perf_counter()
            history = model.fit(
                x_train,
                y_train,
                validation_data=(x_validation, y_validation),
                epochs=MAX_EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=True,
                callbacks=callbacks,
                verbose=0,
            )
            test_loss, test_acc = model.evaluate(
                x_test,
                y_test,
                batch_size=BATCH_SIZE,
                verbose=0,
            )
            training_time = time.perf_counter() - start_time
            best_index = int(np.argmin(history.history["val_loss"]))

            new_row = {
                "dataset": DATASET_NAME,
                "worker_id": args.worker_id,
                "architecture_index": int(config.architecture_index),
                "architecture_id": str(config.architecture_id),
                "model_name": str(config.model_name),
                "seed": int(seed),
                "epochs_completed": int(len(history.history["loss"])),
                "best_epoch": best_index + 1,
                "best_val_loss": float(history.history["val_loss"][best_index]),
                "best_val_acc": float(history.history["val_accuracy"][best_index]),
                "test_loss": float(test_loss),
                "test_acc": float(test_acc),
                "num_parameters_total": int(model.count_params()),
                "training_time_s": float(training_time),
            }
            runs = pd.concat([runs, pd.DataFrame([new_row])], ignore_index=True)
            runs = runs.sort_values(["architecture_index", "seed"])
            atomic_write_csv(runs, runs_path)
            summary = make_summary(runs, worker_architectures)
            atomic_write_csv(summary, summary_path)
            completed.add(key)

            print(
                f"Finished {config.architecture_id}, seed={seed}: "
                f"test_acc={test_acc:.4f}, best_epoch={best_index + 1}, "
                f"time={training_time / 60.0:.1f} min"
            )

            del model, history
            tf.keras.backend.clear_session()
            gc.collect()

            new_runs_completed += 1
            if (
                args.max_new_runs is not None
                and new_runs_completed >= args.max_new_runs
            ):
                print(
                    f"Stopped after {new_runs_completed} new run(s) as requested. "
                    "Run again without --max-new-runs to resume."
                )
                return

    print(
        f"{DATASET_DISPLAY_NAME} worker {args.worker_id} finished all "
        "assigned runs."
    )
    print(f"Runs: {runs_path}")
    print(f"Summary: {summary_path}")


def main():
    global DATA_DIR
    args = parse_args()
    DATA_DIR = Path(args.data_dir).resolve()
    study_dir = Path(args.study_dir).resolve()
    study_dir.mkdir(parents=True, exist_ok=True)

    if args.prepare_only:
        prepare_assignments(study_dir, args.pilot_size, args.workers, args.metrics_file)
        return

    train_worker(args, study_dir)


if __name__ == "__main__":
    main()
