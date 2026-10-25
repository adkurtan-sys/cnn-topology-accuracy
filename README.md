# CNN topology and test accuracy

Supporting code and architecture-level data for **Predicting the Accuracy of
Convolutional Neural Networks Using Topological Metrics of Neuron-Level Graphs**.

## Contents

| Folder | Contents |
| --- | --- |
| `analysis/` | R analysis and six original Excel files: 162 development and 54 holdout architectures per dataset |
| `training/` | CIFAR-100, CIFAR-10 and SVHN training code, shared data reader, aggregation utility and original architecture/metric CSV |
| `benchmark/` | Standalone mapping/three-metric timing experiment and detailed instructions |
| `benchmark/results/` | Note explaining the scope of the published timing material |
| `notebooks/` | Unchanged historical Paperspace notebook, not a portable Run All workflow |
| `tests/` | Lightweight tests; no image download, GPU training or graph allocation |

The model family has fixed depth, filters 16/32/64/128 and strides 1/1/2/1.
Conv0 is standard 3x3. In Conv1-3, convolution groups 1/2/4 and kernels 3/5
vary independently: 216 architectures. Graphs cover input through Conv3 only.

## Reproducibility scope

The Excel files support replication of final regression fits and holdout
results without CNN training. The R script also performs a **new** 20x5-fold
CV check with documented seeds; it is not a reconstruction of the unknown
historical CV folds or every candidate model considered during development.

Training scripts are adapted for portable paths. The original raw 1080-run
CSV collections per dataset, trained CNN weights, historical training
environment metadata and full metric-generation/modularity-search pipeline
are not included in this package. The original metric CSV is supplied as an
input. The benchmark computes only the three final topological predictors.
No claim is made that the entire original exploratory workflow is reproduced.

The benchmark code is provided for new measurements. Original per-architecture
timing outputs and historical Python/GPU environment metadata are not included
in this release. Runtime depends on hardware, software and system load; a new
timing run is not expected to reproduce the exact times reported in the paper.
No code/data reuse license has been assigned to this release. Third-party
datasets and the external graph mapper retain their respective terms.

## 1. Repeat the statistical analysis

Keep `analiza.R` and all six Excel files together in `analysis/`.
Install R and its packages once:

```r
install.packages(c("readxl", "lmtest", "sandwich"))
```

From the repository root:

```sh
Rscript analysis/analiza.R
```

Or open `analysis/analiza.R` in RStudio and use Source. Each run creates a new
`analysis/rezultati/analiza_.../` directory; existing inputs are not overwritten.
Outputs include coefficients with HC3 uncertainty, holdout predictions and
metrics, diagnostics, serialized regressions, CV results and `sessionInfo.txt`.
CSV exports use semicolon delimiters and decimal commas (`read.csv2` in R).
The descriptive table is Supplementary Table S1, not the manuscript's timing
Table 1. Accuracy is on the 0-1 scale; percentage-point errors multiply by 100.

Expected rounded holdout results:

| Dataset | Predictive R2 | Spearman rho | RMSE (percentage points) |
| --- | ---: | ---: | ---: |
| CIFAR-100 | 0.839 | 0.903 | 0.606 |
| CIFAR-10 | 0.877 | 0.928 | 0.726 |
| SVHN | 0.854 | 0.910 | 0.853 |

Coefficients are fitted separately per dataset. The same 54 architecture IDs
are held out throughout; they must not be used to tune the model.

## 2. Python environment and input images

Use an isolated Python environment. Install the packages listed in
`requirements.txt`, choosing TensorFlow/GPU dependencies for your platform.
These unpinned names are NOT the original environment lock file. Record
`python --version` and `python -m pip freeze` for new runs. GPU retraining is
expensive and numerical results can vary with software and hardware.

Download the official **Python archives** of CIFAR-10 and CIFAR-100 from the
[CIFAR dataset page](https://www.cs.toronto.edu/~kriz/cifar.html).
Verify the archive checksums listed by the provider and extract them locally.
For SVHN, obtain the cropped-digit **Format 2** `train_32x32.mat` and
`test_32x32.mat` from the [SVHN source](http://ufldl.stanford.edu/housenumbers/).
The extra split is not used. Images are not redistributed in this package.
Only load official trusted CIFAR pickle files, never unknown pickle inputs.

Expected layout (default `training/datasets/`):

```text
datasets/
  cifar-100-python/train
  cifar-100-python/test
  cifar-10-batches-py/data_batch_1 ... data_batch_5
  cifar-10-batches-py/test_batch
  train_32x32.mat
  test_32x32.mat
```

Use `--data-dir /absolute/path/to/datasets` to override this location. Scripts
do not download images automatically. Missing data cause explicit errors.

## 3. Prepare assignments and train

Run the following once, from the repository root, before starting workers:

```sh
python training/train_cifar100_pilot18.py --prepare-only --pilot-size 216 --workers 10
python training/train_cifar10_full216.py --prepare-only --pilot-size 216 --workers 10
python training/train_svhn_full216.py --prepare-only --pilot-size 216 --workers 10
```

All read `training/graph_metrics_same_216.csv` by default and write assignments
to `training/runs/`. Override with `--metrics-file` and `--study-dir` if needed.
Despite its historical name, `train_cifar100_pilot18.py` handles all 216 when
`--pilot-size 216` is supplied. Keep all Python files in `training/` together:
SVHN imports the CIFAR-10 engine, and CIFAR-100 imports `dataset_io.py`.

Example for worker 1 (choose the dataset you intend to train):

```sh
python training/train_cifar100_pilot18.py --worker-id 1 --workers 10 --pilot-size 216
python training/train_cifar10_full216.py --worker-id 1 --workers 10 --pilot-size 216
python training/train_svhn_full216.py --worker-id 1 --workers 10 --pilot-size 216
```

Repeat for worker IDs 2 through 10, using identical assignments. Workers may
run on separate machines or sequentially on one machine. Never run the same
dataset/worker twice concurrently. For separate storage, copy the assignment
files to every worker's study directory, then collect its outputs into the
corresponding `training_<dataset>_full216/` directory. Avoid changing worker
counts or assignments when resuming. Start new experiments in an empty
`--study-dir`, using the same option for preparation, training and aggregation.

`--max-new-runs 1` permits a one-training smoke test. Repeating the command
without that flag resumes remaining architecture/seed pairs. The CIFAR-100
engine can reuse pilot results found in `training_cifar100_pilot18/`; only
place matching original-protocol results there. A fresh directory avoids reuse.

Training seeds: 11, 22, 33, 44, 55; validation split seed: 123. SVHN subset
seeds: 20260929 (training pool), 20260930 (test). The SVHN split routine differs
from CIFAR to keep exact stratified sample counts. Each dataset uses
45,000 training, 5,000 validation and 10,000 test images, normalized by 255.
Adam 0.001, batch 64, at most 100 epochs; early stopping and LR reduction are
defined in the scripts. `best_epoch` in logs is the absolute validation-loss
minimum, which can differ from the restored epoch under early stopping's
minimum-improvement threshold. Test accuracy uses the actual restored weights.

## 4. Combine new training results

After all workers finish (216 x 5 = 1080 records), run for each dataset:

```sh
python training/merge_runs.py --dataset cifar100 --output training/merged_cifar100
python training/merge_runs.py --dataset cifar10 --output training/merged_cifar10
python training/merge_runs.py --dataset svhn --output training/merged_svhn
```

The utility validates all architecture/seed/worker combinations, aggregates
five test accuracies, and reuses the fixed split and final predictors from the
supplied Excel files. Historical accuracy columns are NOT used in aggregation.
It writes new CSVs, not replacement Excel inputs. For R analysis of a new
experiment, explicitly adapt the input reader or create separately named input
workbooks; do not silently overwrite the published data.

## 5. Mapping and metric timing (no CNN training)

```sh
python benchmark/benchmark_216_topology.py --output benchmark/new_run1
```

This CPU-only experiment needs TensorFlow, NumPy, NetworKit and several GB of
RAM, but no images. It fetches the mapper at a pinned Git commit. See
`benchmark/README_benchmark_216.md` for timing boundaries, smoke test, resume
and output details. Record real new results separately from original timings.
Metric time includes the unweighted graph copy; model construction,
initialization, imports and file output are excluded. Old time columns in the
training CSV cover other metrics and must not be used for manuscript Table 1.

## 6. Lightweight verification

```sh
python -m unittest discover -s tests -v
```

Tests require NumPy, pandas and openpyxl but not TensorFlow, NetworKit or a GPU.
They check Python syntax, architecture/graph sizes, Excel/CSV consistency,
fixed partitions, worker assignments, aggregation and deterministic splitting.
They do not establish GPU-level numerical reproducibility.

## Packaging changes and provenance

Original Excel/CSV data and the notebook were copied without edits. See
`source_manifest.csv` for their SHA-256 hashes. Scripts retain the study's
model construction, seeds, optimization and callbacks. Changes are limited to
portable paths, a direct official-CIFAR-100 reader instead of cache symlink
replacement, and an explicit data path in metadata. No GPU training has been
performed while assembling this release. New helper scripts/tests are identified
by their purpose above. R calculations are unchanged; only the descriptive-table
filename/console label were changed to S1. See `VALIDATION.md` for actual checks.
