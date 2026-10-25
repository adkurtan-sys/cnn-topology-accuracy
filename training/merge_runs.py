"""Validate new worker CSVs and aggregate them using the published split.

Never overwrites the original Excel inputs. Reads only architecture IDs and
topological predictors from them, not their historical accuracy values.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (11, 22, 33, 44, 55)
TABLES = {
    "cifar100": ("cifar100_development162_modularity5.xlsx", "development_162",
                 "cifar100_final_holdout54.xlsx", "final_holdout_54"),
    "cifar10": ("cifar10_development162.xlsx", "Sheet1",
                "cifar10_final_holdout54.xlsx", "Sheet1"),
    "svhn": ("svhn_development162.xlsx", "development_162",
             "svhn_final_holdout54.xlsx", "final_holdout_54"),
}


def reference_table(dataset):
    dev, ds, hold, hs = TABLES[dataset]
    cols = ["architecture_id", "avg_degree", "normalized_degree_entropy",
            "assortativity_coefficient"]
    parts = []
    for filename, sheet, split, count in (
            (dev, ds, "development", 162), (hold, hs, "final_holdout", 54)):
        frame = pd.read_excel(ROOT / "analysis" / filename, sheet_name=sheet, usecols=cols)
        if len(frame) != count or frame.architecture_id.duplicated().any():
            raise ValueError(f"Invalid reference split: {filename}")
        frame["analysis_split"] = split
        parts.append(frame)
    reference = pd.concat(parts, ignore_index=True)
    if reference.architecture_id.duplicated().any():
        raise ValueError("Development/holdout overlap")
    return reference


def aggregate(runs, assignments, reference, dataset):
    if len(assignments) != 216 or assignments.architecture_id.duplicated().any():
        raise ValueError("Expected 216 unique assignments")
    required = ["architecture_id", "worker_id", "seed", "test_acc", "test_loss"]
    if any(c not in runs for c in required) or runs[required].isna().any().any():
        raise ValueError("Missing run columns/values")
    if len(runs) != 1080 or runs.duplicated(["architecture_id", "seed"]).any():
        raise ValueError("Expected exactly 1080 unique architecture/seed runs")
    if "dataset" in runs and not runs.dataset.eq(dataset).all():
        raise ValueError("Mixed or incorrect dataset")
    if not np.isfinite(runs[["test_acc", "test_loss"]].to_numpy()).all():
        raise ValueError("Non-finite training results")
    if not runs.test_acc.between(0, 1).all() or not runs.test_loss.ge(0).all():
        raise ValueError("Invalid accuracy/loss")
    expected = {(r.architecture_id, int(r.worker_id), s)
                for r in assignments.itertuples() for s in SEEDS}
    actual = set(runs[["architecture_id", "worker_id", "seed"]].itertuples(index=False, name=None))
    if actual != expected or set(assignments.architecture_id) != set(reference.architecture_id):
        raise ValueError("Missing/unexpected runs or architecture IDs")
    summary = runs.groupby("architecture_id", as_index=False).agg(
        completed_seeds=("seed", "nunique"), mean_test_acc=("test_acc", "mean"),
        std_test_acc=("test_acc", "std"), mean_test_loss=("test_loss", "mean"))
    return reference.merge(summary, on="architecture_id", validate="one_to_one")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=TABLES, required=True)
    parser.add_argument("--study-dir", type=Path, default=Path(__file__).resolve().parent / "runs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = sorted((args.study_dir / f"training_{args.dataset}_full216").glob("worker_*_runs.csv"))
    if not files:
        raise FileNotFoundError("No worker CSVs found")
    runs = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    assignments = pd.read_csv(args.study_dir / f"{args.dataset}_full216_assignments.csv")
    full = aggregate(runs, assignments, reference_table(args.dataset), args.dataset)
    args.output.mkdir(parents=True, exist_ok=False)
    full.to_csv(args.output / "full216.csv", index=False)
    for split in ("development", "final_holdout"):
        full.loc[full.analysis_split.eq(split)].to_csv(args.output / f"{split}.csv", index=False)
    runs.to_csv(args.output / "all_1080_runs.csv", index=False)
    print(f"Validated 216 architectures x 5 seeds; saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
