# Data And Cache Layout

This project keeps reusable numerical results in `data/`. Notebooks should not
write reusable CSV, JSON, or figure files under `notebooks/data/`; use the helper
functions in `src.data.paths` instead.

## Directory Policy

- `data/<molecule>/<basis>/fci_*.csv`: cached FCI or CASCI energy curves.
- `data/<molecule>/<basis>/fci_*.json`: metadata paired with each FCI/CASCI CSV.
- `data/<molecule>/<basis>/vqe_cache/`: cached VQE experiment outputs.
- `data/<molecule>/<basis>/figures/`: generated figures tied to one dataset.
- `data/<molecule>/<basis>/reference/`: curated or literature-derived reusable data.
- `data/noise_cache/`: cache files for noise-model experiments that are not tied
  to a single molecule/basis directory.

## Path Helpers

Use these helpers instead of manually assembling paths in notebooks:

```python
from src.data.paths import (
    get_fci_cache_dir,
    get_figure_dir,
    get_noise_cache_dir,
    get_reference_data_dir,
    get_vqe_cache_dir,
)
# For example:

reference_dir = get_reference_data_dir("LiH", "sto-3g", create=True)
vqe_dir = get_vqe_cache_dir("LiH", "sto-3g", create=True)
noise_dir = get_noise_cache_dir(create=True)
```

The intent is that `data/` can be reused by other researchers without having to
inspect notebook working directories or guess which CSV files are canonical.

## Inventory

Use `summarize_data_dir` to inspect the cache programmatically:

```python
import pandas as pd
from src.data import summarize_data_dir

pd.DataFrame(summarize_data_dir())
```
