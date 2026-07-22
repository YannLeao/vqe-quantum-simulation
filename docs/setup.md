# Setup Guide

This project depends on PySCF, Qiskit Nature, and scientific Python packages
that are most reliable on Linux. If you use Windows, run the project through
WSL2 with Ubuntu.

## Recommended Platform

- Linux or WSL2 Ubuntu
- Conda or Miniconda
- Python 3.11
- JupyterLab or PyCharm Professional for notebooks

## Conda Environment

From the repository root:

```bash
conda env create -f environment.yml
conda activate quantum
```

To update an existing environment:

```bash
conda env update -f environment.yml --prune
conda activate quantum
```

The environment installs the repository in editable mode:

```bash
pip install -e .[dev]
```

This allows notebooks to import modules from `src/` while local code changes are
picked up immediately.

## WSL2 Notes

On Windows, open Ubuntu/WSL and verify the environment:

```bash
conda activate quantum
python -c "import pyscf; print(pyscf.__version__)"
python -c "import qiskit; print(qiskit.__version__)"
```

If PyCharm is used, configure the interpreter as the Python executable inside
the WSL Conda environment, for example:

```text
/home/<user>/miniconda3/envs/quantum/bin/python
```

When possible, start Jupyter from WSL rather than from native Windows:

```bash
cd /mnt/c/Users/<windows-user>/PycharmProjects/vqe-quantum-simulation
conda activate quantum
jupyter lab
```

## Environment Variables

IBM Quantum credentials are optional and are not required for the final
statevector notebooks.

For experiments that access IBM services, create a local `.env` file based on
`.env.example`. Never commit `.env`.

## Common Issues

- If PySCF fails on native Windows, use WSL2.
- If notebooks cannot import `src`, update the Conda environment or run
  `pip install -e .[dev]` from the repository root.
- If PyCharm gets stuck indexing or cleaning skeletons, restart the IDE after
  confirming that the WSL interpreter can import PySCF from the terminal.
