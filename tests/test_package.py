"""Offline package checks: no TensorFlow, GPU training or dataset downloads."""
import ast
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "benchmark"))
import benchmark_216_topology as bench
import merge_runs
import train_cifar100_pilot18 as c100
import train_svhn_full216 as svhn
from dataset_io import load_cifar100


class PackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metrics = pd.read_csv(ROOT / "training" / "graph_metrics_same_216.csv")

    def test_syntax(self):
        for path in ROOT.rglob("*.py"):
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))

    def test_all_architectures_and_graph_counts(self):
        source = self.metrics.set_index("architecture_id")
        configs = bench.configurations()
        self.assertEqual(len(configs), 216)
        self.assertEqual(len({c["architecture_id"] for c in configs}), 216)
        for config in configs:
            row = source.loc[config["architecture_id"]]
            for key, prefix in (("kernels", "kernel"), ("groups", "groups"),
                                ("filters", "filters"), ("strides", "stride")):
                self.assertEqual(config[key], [int(row[f"{prefix}{i}"]) for i in range(4)])
            nodes, edges = bench.expected_size(config)
            self.assertEqual(nodes, int(row.num_nodes))
            self.assertEqual(edges, int(row.num_edges))
            self.assertAlmostEqual(2 * edges / nodes, row.avg_degree)

    def test_excel_inputs_and_shared_partition(self):
        common_ids = None
        predictors = ["avg_degree", "normalized_degree_entropy", "assortativity_coefficient"]
        for dataset in merge_runs.TABLES:
            ref = merge_runs.reference_table(dataset)
            ids = set(ref.loc[ref.analysis_split.eq("final_holdout"), "architecture_id"])
            self.assertEqual(len(ids), 54)
            if common_ids is None:
                common_ids = ids
            self.assertEqual(ids, common_ids)
            joined = ref.merge(self.metrics, on="architecture_id", suffixes=("_excel", "_csv"))
            self.assertEqual(len(joined), 216)
            for col in predictors:
                np.testing.assert_allclose(joined[col + "_excel"], joined[col + "_csv"], rtol=1e-9, atol=1e-12)

    def test_cli_preparation_and_help(self):
        for script, dataset in (("train_cifar100_pilot18.py", "cifar100"),
                                ("train_cifar10_full216.py", "cifar10"),
                                ("train_svhn_full216.py", "svhn")):
            command = [sys.executable, str(ROOT / "training" / script)]
            subprocess.run(command + ["--help"], check=True, capture_output=True)
            with tempfile.TemporaryDirectory() as tmp:
                subprocess.run(command + ["--prepare-only", "--pilot-size", "216",
                               "--workers", "10", "--study-dir", tmp], check=True, capture_output=True)
                frame = pd.read_csv(Path(tmp) / f"{dataset}_full216_assignments.csv")
                self.assertEqual(len(frame), 216)
                self.assertEqual(frame.architecture_id.nunique(), 216)
                self.assertEqual(sorted(frame.groupby("worker_id").size()), [21]*4 + [22]*6)

    def test_cifar_split(self):
        labels = np.repeat(np.arange(100), 500)
        train, val = c100.stratified_split(labels, .1, 123)
        self.assertEqual(len(train), 45000)
        self.assertEqual(len(val), 5000)
        self.assertFalse(set(train) & set(val))
        again = c100.stratified_split(labels, .1, 123)
        np.testing.assert_array_equal(train, again[0])
        np.testing.assert_array_equal(val, again[1])

    def test_svhn_sampling(self):
        labels = np.arange(73257) % 10
        idx = svhn.stratified_sample_indices(labels, 50000, 20260929)
        self.assertEqual(len(set(idx)), 50000)
        np.testing.assert_array_equal(idx, svhn.stratified_sample_indices(labels, 50000, 20260929))
        train, val = svhn.stratified_split_exact(labels[idx], .1, 123)
        self.assertEqual((len(train), len(val)), (45000, 5000))
        self.assertFalse(set(train) & set(val))

    def test_aggregate_and_invalid_runs(self):
        ref = merge_runs.reference_table("cifar100")
        assignments = pd.DataFrame({"architecture_id": ref.architecture_id, "worker_id": 1})
        runs = pd.DataFrame([
            dict(architecture_id=a, worker_id=1, seed=s, test_acc=.5, test_loss=1.)
            for a in ref.architecture_id for s in merge_runs.SEEDS])
        full = merge_runs.aggregate(runs, assignments, ref, "cifar100")
        self.assertEqual(len(full), 216)
        self.assertTrue(full.mean_test_acc.eq(.5).all())
        for invalid in (runs.iloc[:-1], pd.concat([runs.iloc[:-1], runs.iloc[:1]])):
            with self.assertRaises(ValueError):
                merge_runs.aggregate(invalid, assignments, ref, "cifar100")
        bad = runs.copy()
        bad.loc[0, "test_acc"] = np.inf
        with self.assertRaises(ValueError):
            merge_runs.aggregate(bad, assignments, ref, "cifar100")

    def test_cifar100_reader(self):
        # Synthetic trusted batches test channel order without allocating the dataset.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cifar-100-python"
            root.mkdir()
            for name in ("train", "test"):
                (root / name).touch()
            batch = {b"data": np.zeros((50000, 3072), dtype=np.uint8),
                     b"fine_labels": np.arange(50000) % 100}
            batch[b"data"][0, 1024] = 73
            test = {b"data": batch[b"data"][:10000], b"fine_labels": batch[b"fine_labels"][:10000]}
            with patch("dataset_io.pickle.load", side_effect=[batch, test]):
                (x, y), (xt, yt) = load_cifar100(tmp)
            self.assertEqual(x.shape, (50000, 32, 32, 3))
            self.assertEqual(x[0, 0, 0, 1], 73)
            self.assertEqual(y[101], 1)
            self.assertEqual(len(yt), 10000)


if __name__ == "__main__":
    unittest.main()
