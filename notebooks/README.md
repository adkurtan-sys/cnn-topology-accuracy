# Historical notebook

`Rad216-1.ipynb` is preserved unchanged, including original saved outputs.
It documents worker 1, checks of combined worker results, the architecture
split procedure, and historical exports. It is NOT a portable Run All entry
point: it refers to original `/storage` and `/notebooks` paths and additional
pilot assignments, modularity tables, raw worker results and metadata.

Use the main README and packaged training scripts for new runs. Use
`training/merge_runs.py` to combine new worker outputs without overwriting the
published Excel inputs. The published 162/54 architecture partition is read
from those inputs, rather than re-selected. No need to rerun the historical
50,000-candidate partition search or obtain ten copies of the notebook.
