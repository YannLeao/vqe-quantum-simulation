import csv
import time
from pathlib import Path
from typing import Callable

import numpy as np

from src.solvers.fci import compute_fci_energy
from src.utils.paths import get_project_root


def cache_fci(
        molecule: str,
        geometry_fn: Callable[[float], str],
        distances: np.ndarray = np.linspace(0.3, 2.5, 100),
        basis: str = "sto-3g",
        save: bool = True,
        data_dir: Path | None = None,
        verbose: bool = True,
):
    """
    Compute and cache FCI energies for a molecule over a range of distances.

    Parameters
    ----------
    molecule : str
        Name of the molecule (used for folder structure).
    geometry_fn : Callable
        Function that takes a distance (float) and returns an atom string.
    distances : np.ndarray
        Array of distances.
    basis : str
        Basis set.
    save : bool
        Whether to save results to disk.
    data_dir : Path
        Base directory for saving data.
    verbose : bool
        Print progress.

    Returns
    -------
    np.ndarray
        Array of computed energies.
    """

    if data_dir is None:
        data_dir = get_project_root() / "data"

    path = data_dir / molecule / basis
    path.mkdir(parents=True, exist_ok=True)

    file_path = path / "fci.csv"

    results = []
    done = {}

    if file_path.exists():
        with open(file_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                d = float(row["distance"])
                e = float(row["fci_energy"])
                done[d] = e

    if save and not file_path.exists():
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["distance", "fci_energy"])

    start_time = time.time()

    if save:
        f = open(file_path, "a", newline="")
        writer = csv.writer(f)

    try:
        for i, d in enumerate(distances):
            if any(abs(d - x) < 1e-8 for x in done): # if done in done:
                results.append(done[d])
                if verbose:
                    print(f"[{i+1}/{len(distances)}] d={d:.4f} (cached)")
                continue

            try:
                atom_string = geometry_fn(d)

                energy = compute_fci_energy(atom_string, basis)

                if energy is None:
                    raise ValueError("FCI energy returned None")

                results.append(energy)

                if save:
                    writer.writerow([d, energy])
                    f.flush()

                if verbose:
                    elapsed = time.time() - start_time
                    print(
                        f"[{i+1}/{len(distances)}] d={d:.4f} "
                        f"E={energy:.8f} ⏱ {elapsed:.1f}s"
                    )

            except Exception as e:
                print(f"Erro em d={d:.4f}: {e}")
                results.append(np.nan)

    finally:
        if save:
            f.close()

    return np.array(results)