# VQE Quantum Simulation

This repository contains computational experiments developed during an undergraduate
research project (PIBIC/FACEPE) focused on quantum simulation methods applied to
small molecular systems.

The current codebase combines reusable Python modules, cached FCI/VQE results,
research notebooks, and notes used to study Fourier-based strategies for VQE.

## Repository Structure

- `src/`: reusable implementation for VQE, FCI, data cache, plotting, and utilities.
- `data/`: cached numerical results used by the notebooks and experiments.
- `notebooks/`: exploratory and reproducible analyses.
- `notes/`: PDF notes and research references.
- `outputs/`: generated figures, logs, and other local outputs.

## Setup

This project uses Conda because some scientific dependencies, especially PySCF,
are easier to install from `conda-forge`.

```bash
conda env create -f environment.yml
conda activate quantum
```

If the environment already exists, update it with:

```bash
conda env update -f environment.yml --prune
conda activate quantum
```

The Conda environment installs this repository in editable mode, so imports such
as `from src.data import cache_fci` work from notebooks and scripts while local
changes in `src/` are picked up immediately.

## Running Notebooks

Start Jupyter from the repository root:

```bash
jupyter lab
```

Suggested reading order:

1. `notebooks/00_fci_cache_baseline.ipynb`: baseline cache workflow for FCI data.
2. `notebooks/01_fourier_analysis.ipynb`: exploratory Fourier analysis.
3. `notebooks/02_fourier_tradeoff_article_plan.ipynb`: article/tradeoff planning experiments.

## Notes On Data

The `data/` directory currently contains generated caches. Most CSV files have a
paired JSON metadata file describing the experiment configuration that produced
them. Future refactors should preserve this metadata-first idea and avoid relying
only on long file names to understand a dataset.

## Research Reports

Project reports and documentation are maintained in separate repositories:

- Partial Report: [partial_report.pdf](https://github.com/YannLeao/vqe-research-report/blob/main/partial/report.pdf)
- Final Report: to be added.
