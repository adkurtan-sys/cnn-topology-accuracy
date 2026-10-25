"""Map all 216 study backbones and compute three topology metrics; NO training.

Use the existing TensorFlow/NumPy/NetworKit environment. Run as a script in a
fresh process. No image dataset, Excel inputs or GPU is required. Graphs can
need several GB of system RAM. One measurement per architecture, sequentially.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import urllib.request

SHA = "f1cfec7348f46b202543db2c8030af155cfc1d2a"
URL = f"https://raw.githubusercontent.com/adkurtan-sys/keras-cnn-to-graph/{SHA}/keras_cnn_to_graph.py"
SEED = 11


def configurations():
    options = list(product(("std", "group2", "group4"), (3, 5)))
    groups = {"std": 1, "group2": 2, "group4": 4}
    return [dict(architecture_id=f"A{i:03d}", padding="same",
                 filters=[16, 32, 64, 128], strides=[1, 1, 2, 1],
                 kernels=[3] + [k for _, k in layers],
                 groups=[1] + [groups[c] for c, _ in layers])
            for i, layers in enumerate(product(options, repeat=3), 1)]


def expected_size(config):
    """Independent count of real connections, excluding SAME padding values."""
    height, channels, nodes, edges = 32, 3, 32 * 32 * 3, 0
    for filters, kernel, stride, groups in zip(
            config["filters"], config["kernels"], config["strides"], config["groups"]):
        output = math.ceil(height / stride)
        before = max((output - 1) * stride + kernel - height, 0) // 2
        valid = sum(0 <= i * stride - before + j < height
                    for i in range(output) for j in range(kernel))
        edges += valid * valid * filters * (channels // groups)
        nodes += output * output * filters
        height, channels = output, filters
    return nodes, edges


def build_backbone(tf, config):
    inputs = tf.keras.Input(shape=(32, 32, 3), name="input")
    x = inputs
    for i in range(4):
        x = tf.keras.layers.Conv2D(
            config["filters"][i], config["kernels"][i],
            strides=config["strides"][i], groups=config["groups"][i],
            padding="same", activation="relu", use_bias=True, name=f"conv{i}")(x)
    return tf.keras.Model(inputs, x, name=config["architecture_id"].lower())


def three_metrics(graph, nk, np):
    if graph.isDirected():
        raise ValueError("Expected undirected graph")
    u = nk.graphtools.toUnweighted(graph) if graph.isWeighted() else graph
    n, m = u.numberOfNodes(), u.numberOfEdges()
    if not n or not m or u.upperNodeIdBound() != n or u.numberOfSelfLoops():
        raise ValueError("Expected contiguous, loop-free, nonempty graph")
    k = np.fromiter((u.degree(i) for i in range(n)), dtype=float, count=n)
    if not np.isclose(k.sum(), 2*m):
        raise ValueError("Degree sum differs from 2M")
    _, counts = np.unique(k, return_counts=True)
    p = counts / n
    h = float(-np.sum(p * np.log2(p)))
    hn = h / math.log2(len(counts)) if len(counts) > 1 else 0.0
    endpoint_mean = np.dot(k, k) / (2*m)
    if np.dot(k, (k-endpoint_mean)**2) <= 0:
        raise ValueError("Undefined assortativity")
    algo = nk.correlation.Assortativity(u, k.tolist())
    algo.run()
    r = float(algo.getCoefficient())
    if not math.isfinite(r) or abs(r) > 1 + 1e-10:
        raise ValueError("Invalid assortativity")
    return dict(num_nodes=n, num_edges=m, avg_degree=float(k.mean()),
                normalized_degree_entropy=hn, assortativity_coefficient=r)


def dump_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def dump_csv(path, rows):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    tmp.replace(path)


def metadata(tf, nk, np):
    cpu = platform.processor()
    ram = None
    if Path("/proc/cpuinfo").exists():
        cpu = next((line.split(":", 1)[1].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), cpu)
    if Path("/proc/meminfo").exists():
        ram = next((line for line in Path("/proc/meminfo").read_text().splitlines()
                    if line.startswith("MemTotal:")), None)
    return dict(python=platform.python_version(), tensorflow=tf.__version__,
                numpy=np.__version__, networkit=nk.__version__, cpu=cpu, ram=ram,
                platform=platform.platform(), logical_cpus=os.cpu_count(),
                cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                networkit_threads=nk.getMaxNumberOfThreads(),
                tf_intra_threads=tf.config.threading.get_intra_op_parallelism_threads(),
                tf_inter_threads=tf.config.threading.get_inter_op_parallelism_threads())


def summarize(rows, np):
    return [dict(stage=label, n=len(rows), complete_216=len(rows) == 216,
                 total_s=float(sum(values)), mean_s=float(np.mean(values)),
                 sd_s=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                 median_s=float(np.median(values)), min_s=min(values), max_s=max(values))
            for key, label in (("mapping_s", "Graph mapping"),
                               ("metrics_s", "Three metrics"),
                               ("topology_s", "Mapping and metrics"))
            for values in [[row[key] for row in rows]]]


def run(args):
    # Run on CPU; no image dataset or GPU initialization is needed.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import numpy as np
    import networkit as nk
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    configs = configurations()
    if args.smoke_test:
        configs = configs[:1]
    if args.resume and args.output is None:
        raise ValueError("--resume requires --output")
    out = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix="topology216_", dir=Path.cwd()))
    out.mkdir(parents=True, exist_ok=True)
    fingerprint = dict(configurations=configs, seed=SEED, mapper_commit=SHA,
                       script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       environment=metadata(tf, nk, np), smoke_test=args.smoke_test)
    lock = out / "RUNNING.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"Lock exists: {lock}. Check whether the earlier process is still running.")
    os.close(fd)
    try:
        if args.resume:
            saved = json.loads((out / "metadata.json").read_text())
            if any(saved.get(k) != v for k, v in fingerprint.items()):
                raise ValueError("Script, configuration or environment changed; start a new run")
            source = (out / "mapper.py").read_bytes()
            if hashlib.sha256(source).hexdigest() != saved["mapper_sha256"]:
                raise ValueError("Mapper checksum mismatch")
        else:
            if any(p.name != lock.name for p in out.iterdir()):
                raise ValueError("Output is not empty; use --resume or a new output folder")
            print("Downloading pinned GitHub mapper (outside timers)...", flush=True)
            with urllib.request.urlopen(URL, timeout=120) as response:
                source = response.read()
            compile(source, "mapper.py", "exec")
            (out / "mapper.py").write_bytes(source)
            dump_json(out / "metadata.json", dict(
                fingerprint, mapper_url=URL, mapper_sha256=hashlib.sha256(source).hexdigest(),
                scope="Input through Conv3 only; undirected weighted graph; unweighted degree metrics.",
                timing="One measurement per architecture, sequentially. No warm-up excluded. "
                       "Mapping timer covers the public mapper call including its checks. "
                       "Metric timer includes unweighted copy, degree extraction, metrics and their checks. "
                       "Imports, model construction, downloads, file I/O and final cleanup are excluded. "
                       "Total is a sum of measured stages, not complete notebook wall time. "
                       "SD describes differences across architectures, not repeated-run uncertainty."))
        namespace = {"__name__": "pinned_mapper", "__file__": str(out / "mapper.py")}
        exec(compile(source, str(out / "mapper.py"), "exec"), namespace)
        mapper = namespace["keras_cnn_to_graph"]
        rows = []
        print("Output:", out, "\nArchitectures:", len(configs), flush=True)
        for i, cfg in enumerate(configs, 1):
            a = cfg["architecture_id"]
            checkpoint = out / f"{a}.json"
            if checkpoint.exists():
                row = json.loads(checkpoint.read_text())
                if row["architecture_id"] != a or (row["num_nodes"], row["num_edges"]) != expected_size(cfg):
                    raise ValueError(f"Invalid checkpoint: {a}")
                rows.append(row)
                print(f"[{i}/{len(configs)}] {a}: already complete", flush=True)
                continue
            tf.keras.backend.clear_session(); gc.collect()
            tf.keras.utils.set_random_seed(SEED)
            model = build_backbone(tf, cfg)
            for w in model.weights:
                w.numpy()
            print(f"[{i}/{len(configs)}] {a}: mapping...", flush=True)
            start = time.perf_counter()
            graph, node_to_layer, layer_nodes = mapper(model, directed=False, show_summary=False)
            mapped = time.perf_counter()
            metrics = three_metrics(graph, nk, np)
            finished = time.perf_counter()
            if (metrics["num_nodes"], metrics["num_edges"]) != expected_size(cfg):
                raise ValueError(f"Node/edge count differs from independent CNN count: {a}")
            row = dict(architecture_id=a,
                       **{f"groups{j}": cfg["groups"][j] for j in range(4)},
                       **{f"kernel{j}": cfg["kernels"][j] for j in range(4)},
                       **metrics, mapping_s=mapped-start, metrics_s=finished-mapped,
                       topology_s=finished-start)
            dump_json(checkpoint, row)
            rows.append(row)
            dump_csv(out / "per_architecture.csv", rows)
            dump_csv(out / "timing_summary.csv", summarize(rows, np))
            print(f"  Mapping: {row['mapping_s']:.3f} s; metrics: {row['metrics_s']:.3f} s; "
                  f"total: {row['topology_s']:.3f} s; edges: {row['num_edges']:,}", flush=True)
            del graph, node_to_layer, layer_nodes, model
            gc.collect()
        summary = summarize(rows, np)
        dump_csv(out / "per_architecture.csv", rows)
        dump_csv(out / "timing_summary.csv", summary)
        dump_json(out / "summary.json", summary)
        print("\nSUMMARY (seconds):", flush=True)
        for row in summary:
            sd = f"{row['sd_s']:.3f}" if row["sd_s"] is not None else "n/a"
            print(f"{row['stage']}: N={row['n']}; mean={row['mean_s']:.3f}; SD={sd}; "
                  f"range={row['min_s']:.3f}..{row['max_s']:.3f}; total={row['total_s']:.3f}", flush=True)
        print("Complete:", out, flush=True)
    finally:
        lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="A001 only, for environment checking")
    run(parser.parse_args())
