# VQE Quantum Simulation

Computational research repository developed during an undergraduate PIBIC/FACEPE
project on variational quantum algorithms for molecular simulation.

The project implements a reproducible Python workflow for building electronic
Hamiltonians, running VQE experiments, caching FCI/CASCI references, analyzing
Fourier-inspired energy landscapes, and testing machine-learning models as
configuration recommenders for VQE.

## Project Status

This repository is preserved as a research portfolio and reproducibility
artifact. The final report, LaTeX sources, and paper drafts are maintained in a
separate [writing repository](https://github.com/YannLeao/vqe-research-report);
this repository focuses on code, notebooks, cached data, and technical notes.

## Repository Structure

- `src/`: reusable Python modules for molecular systems, Hamiltonians, VQE,
  FCI/CASCI references, Fourier analysis, visualization, data cache, and ML.
- `notebooks/`: main reproducible notebooks used in the project.
- `data/`: reusable cached numerical results and metadata.
- `notes/`: technical PDFs, research notes, and project planning documents.
- `environment.yml`: Conda environment used to run the project on Linux/WSL.
- `pyproject.toml`: editable Python package configuration.

## Main Notebooks

Suggested reading order:

1. `notebooks/00_fci_cache_baseline.ipynb`: builds/cache FCI or CASCI reference curves.
2. `notebooks/01_vqe_grid_search.ipynb`: noiseless VQE grid search with statevector simulation.
3. `notebooks/02_fourier_guided_initialization.ipynb`: Fourier-inspired analysis of 1D energy landscapes.
4. `notebooks/03_vqe_visualization.ipynb`: figures and summaries for the VQE/ML analysis.

The `*_noisy.ipynb` notebooks are preliminary experiments with synthetic noise.
They are kept for traceability, but the final project results focus on
statevector simulations.

## Setup

PySCF is required by the project and is most reliable in Linux. On Windows, the
recommended setup is WSL2 with Ubuntu and Conda.

```bash
git clone https://github.com/YannLeao/vqe-quantum-simulation.git
cd vqe-quantum-simulation

conda env create -f environment.yml
conda activate quantum
```

If the environment already exists:

```bash
conda env update -f environment.yml --prune
conda activate quantum
```

The Conda environment installs the repository in editable mode through
`pip install -e .[dev]`, so changes in `src/` are immediately available from
the notebooks.

For Windows/WSL details, see [docs/setup.md](docs/setup.md).

## Running

Start Jupyter from the repository root:

```bash
jupyter lab
```

The notebooks assume that they are executed from the project root or that the
repository package has been installed in editable mode.

## Data And Cache

The `data/` directory stores reusable experiment outputs. Most CSV files have a
paired JSON metadata file describing the molecular system, basis, mapper,
ansatz, optimizer, seed, and cache configuration.

Reusable data should be written through the helpers in `src.data.paths`, not
directly inside notebook folders. See [data/README.md](data/README.md).

## Research Scope

The final consolidated results cover:

- FCI/CASCI reference curves for small molecular systems.
- VQE grid search over molecules, ansatz choices, optimizers, and circuit depths.
- Chemical-accuracy analysis against FCI/CASCI references.
- Fourier-inspired landscape analysis as an exploratory study.
- Supervised ML models as VQE configuration recommenders.

Noise simulations and IBM backend-inspired experiments were explored during the
project, but they are not part of the final consolidated results.

## Related Writing

Reports and article drafts are versioned separately to keep this repository
focused on executable research code. The project repository remains the source
for the cached data and computational workflows cited by those documents.
