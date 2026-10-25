# Packaging validation

Checked on 25 October 2026. This records packaging checks, not a new CNN study.

## Passed

- Eight offline Python tests (`python -m unittest discover -s tests -v`).
- AST comparison confirmed 18 core model-building, seed, splitting, selection,
  assignment and summary functions are unchanged from the supplied trainers.
- A basic scan found no common OpenAI/GitHub token or private-key patterns in
  scripts, notebook and documentation; this is not an exhaustive privacy audit.
- Python source parsing and help/preparation commands for all three trainers.
- Assignment preparation for 216 unique architectures across 10 workers
  (six workers with 22 architectures and four with 21).
- All 216 benchmark configurations agree with the original CSV, including
  independent node/edge counts and average degree.
- All six Excel files have the expected development/holdout sizes, common
  holdout IDs, and matching final topology predictors against the CSV.
- Deterministic CIFAR splitting and SVHN sampling/splitting on synthetic labels.
- CIFAR-100 reader channel order/labels checked with synthetic batches.
- Aggregation of synthetic 1080-run data, and rejection of missing, duplicated
  and non-finite runs. Synthetic test records are never published as results.
- Complete R analysis executed, including the new 20x5 CV confirmation on all
  three datasets. Holdout R2, Spearman and RMSE match the manuscript:

| Dataset | R2 | Spearman | RMSE (percentage points) |
| --- | ---: | ---: | ---: |
| CIFAR-100 | 0.8389573 | 0.9032590 | 0.6063021 |
| CIFAR-10 | 0.8772573 | 0.9278826 | 0.7255760 |
| SVHN | 0.8538773 | 0.9104250 | 0.8526808 |

The manuscript's rounded CIFAR-100 coefficients and HC3 intervals also match.
Generated R outputs are excluded from the publication copy as requested.
The input-file hashes are recorded in `source_manifest.csv`.

## Check environment (not the historical training environment)

- Windows x64; Python 3.12.14; NumPy 2.3.5; pandas 3.0.1; openpyxl 3.1.5.
- R 4.3.1; readxl 1.4.2; lmtest 0.9-40; sandwich 3.1-3.

## Not verified here

- No real image download, full dataset load, GPU training, actual graph mapping
  or new timing run. TensorFlow, NetworKit and SciPy were unavailable locally.
- No claim of bitwise-identical retraining across TensorFlow/GPU versions.
- The portable CIFAR-100 loader follows the official Python archive format;
  equivalence on complete real image arrays was not exercised locally.
- Original per-run training CSVs and historical environment metadata were not
  provided. Saved notebook output is not an independent audit of those files.
- Original timing outputs are not included in this release; the benchmark
  script enables new measurements rather than retrieval of the original logs.
- Original full exploratory model-selection history and historical CV folds
  are not reproduced by the R script's explicitly labeled new CV check.

## Changes from the supplied scripts

- Configurable input/output paths in training scripts; no hard-coded storage
  requirement and no deletion/replacement of the user's Keras cache.
- `dataset_io.py` reads official CIFAR-100 fine-label batches directly.
- `merge_runs.py` adds validation/aggregation of new runs using the published
  architecture split. It creates new CSVs and never overwrites the Excel data.
- The R descriptive output is labeled S1 instead of Table 1; calculations and
  CV settings were not changed.
- Benchmark source, historical notebook and all original data are unchanged.
