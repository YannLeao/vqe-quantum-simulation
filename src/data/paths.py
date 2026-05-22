from pathlib import Path
from typing import Optional

from src.utils.paths import get_project_root


def get_data_dir(data_dir: Optional[Path] = None, create: bool = False) -> Path:
    """Return the root directory for reusable project data and caches."""
    path = Path(data_dir) if data_dir is not None else get_project_root() / "data"

    if create:
        path.mkdir(parents=True, exist_ok=True)

    return path


def get_molecule_basis_dir(
    molecule: str,
    basis: str,
    data_dir: Optional[Path] = None,
    create: bool = False,
) -> Path:
    """Return the canonical directory for data tied to a molecule and basis."""
    path = get_data_dir(data_dir) / molecule / basis

    if create:
        path.mkdir(parents=True, exist_ok=True)

    return path


def get_fci_cache_dir(
    molecule: str,
    basis: str,
    data_dir: Optional[Path] = None,
    create: bool = False,
) -> Path:
    """Return the directory where FCI cache files are stored."""
    return get_molecule_basis_dir(molecule, basis, data_dir=data_dir, create=create)


def get_vqe_cache_dir(
    molecule: str,
    basis: str,
    data_dir: Optional[Path] = None,
    create: bool = False,
) -> Path:
    """Return the directory where VQE cache files are stored."""
    path = get_molecule_basis_dir(molecule, basis, data_dir=data_dir) / "vqe_cache"

    if create:
        path.mkdir(parents=True, exist_ok=True)

    return path


def get_figure_dir(
    molecule: str,
    basis: str,
    data_dir: Optional[Path] = None,
    create: bool = False,
) -> Path:
    """Return the directory where generated figures for a dataset are stored."""
    path = get_molecule_basis_dir(molecule, basis, data_dir=data_dir) / "figures"

    if create:
        path.mkdir(parents=True, exist_ok=True)

    return path


def get_reference_data_dir(
    molecule: str,
    basis: str,
    data_dir: Optional[Path] = None,
    create: bool = False,
) -> Path:
    """Return the directory for curated or literature-derived reusable data."""
    path = get_molecule_basis_dir(molecule, basis, data_dir=data_dir) / "reference"

    if create:
        path.mkdir(parents=True, exist_ok=True)

    return path


def get_noise_cache_dir(data_dir: Optional[Path] = None, create: bool = False) -> Path:
    """Return the project-level directory for noise experiment caches."""
    path = get_data_dir(data_dir) / "noise_cache"

    if create:
        path.mkdir(parents=True, exist_ok=True)

    return path
