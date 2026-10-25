# Mapping and three topology metrics for 216 CNN backbones

This standalone script recreates the study's complete factorial design:
standard, two-group and four-group convolutions; 3x3 and 5x5 kernels chosen
independently in Conv1, Conv2 and Conv3. Conv0 is fixed (16 filters, 3x3,
stride 1); subsequent filters are 32/64/128 and strides 1/2/1. All use SAME
padding, bias and ReLU; inputs are 32x32 RGB. The backbone ends at Conv3.
No global pooling or classifier is included in the graph.

## What is and is not run

- All 216 backbones are freshly initialized (seed 11), mapped sequentially into
  undirected weighted graphs and checked against independent node/edge counts.
- Three unweighted metrics are computed: mean degree, degree-distribution
  entropy normalized by log2 of the number of observed degree categories,
  and degree assortativity. Zero weights do not remove connections.
- Mapping and metrics have separate timers. Their sum is also reported.
- No CNN training, no regression fitting/prediction, no ranking and no image
  dataset downloads occur. The same backbones apply to all three image datasets
  in the paper, so the mapping is not repeated for each dataset.

## Run on Paperspace

Use the existing environment that successfully ran the single-CNN benchmark,
with TensorFlow, NumPy and NetworKit installed. CPU and several GB of system
RAM are required; no GPU is used. Exact versions and CPU details are recorded
in metadata.json. Internet is required on the first run to fetch the mapper
from commit f1cfec7348f46b202543db2c8030af155cfc1d2a of
https://github.com/adkurtan-sys/keras-cnn-to-graph .

Place benchmark_216_topology.py in /notebooks and execute in a notebook:

```python
import sys
!{sys.executable} -u /notebooks/benchmark_216_topology.py --output /notebooks/topology216_run1
```

No other Python files, Excel files or R installation are needed. The script
stores and checks the downloaded mapper. Candidate models and graphs are built
one at a time; the graph is released before moving to the next architecture.
Do not run competing workloads on the instance while measuring time.

Continue an interrupted run with the same script and environment:

```python
!{sys.executable} -u /notebooks/benchmark_216_topology.py --resume --output /notebooks/topology216_run1
```

Completed architectures are not repeated. If a forcibly terminated process
left RUNNING.lock behind, first confirm that the old process is no longer
running before removing only that lock file. Do not remove the checkpoints.
For a new run, use a new output folder. For an optional one-model environment
test, use --smoke-test with a separate folder; this is not the 216-model result.

## Outputs

- metadata.json: all configurations, versions, CPU/affinity/RAM, seed, mapper
  commit/checksum and precise timing scope.
- A001.json through A216.json: individual checkpoint results.
- per_architecture.csv: configurations, graph sizes, all three metrics and times.
- timing_summary.csv and summary.json: N, completeness, total, mean, sample SD,
  median, minimum and maximum seconds for mapping, metrics and their sum.
- mapper.py: the exact downloaded source used in the run.

Per-architecture CSV and timing summary CSV are refreshed after each completed
architecture. Final summaries should only be reported when complete_216 is true.

## Timing interpretation

One measurement per architecture, not repeated runs of the same architecture.
The SD/range describe variation across architectures. The mapping timer covers
the public mapper call including its internal validation. Metric timing includes
the unweighted graph copy and checks used in calculating the three metrics.
No separate warm-up is excluded. Imports, model construction/initialization,
source downloading, file output and final graph cleanup are outside the timers.
The total is the sum of measured stages, not complete notebook wall time.

These results establish the cost of mapping and metric calculation for the
whole architectural family. Do not divide these aggregate or mean times by the
single earlier training time and call that a measured family-wide speedup.
No model training times are measured by this script.

The three metrics and configuration/node/edge counts can also be checked
against the original study inputs; do not silently replace original data if
an unexpected discrepancy is found.
