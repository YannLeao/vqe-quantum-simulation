import csv
from pathlib import Path

import numpy as np

from src.utils.paths import get_project_root
from src.vqe.hamiltonian import build_qubit_hamiltonian


def cache_fci(
        molecule: str,
        distances: np.ndarray,
        basis: str = "sto-3g",
        save: bool = True,
        data_dir: Path | None = None
):
    if data_dir is None:
        data_dir = get_project_root() / "data"

    results = []

    path = data_dir / molecule / basis
    path.mkdir(parents=True, exist_ok=True)

    file_path = path / "fci.csv"

    if save and not file_path.exists():
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["distance", "fci_energy"])

    if save:
        with open(file_path, "a", newline="") as f:
            writer = csv.writer(f)

            for d in distances:
                atom_string = f"H 0 0 0; H 0 0 {d}"

                data = build_qubit_hamiltonian(
                    atom_string=atom_string,
                    basis=basis
                )

                energy = data["fci_energy"]
                results.append(energy)

                writer.writerow([d, energy])
    else:
        for d in distances:
            atom_string = f"H 0 0 0; H 0 0 {d}"

            data = build_qubit_hamiltonian(
                atom_string=atom_string,
                basis=basis
            )

            energy = data["fci_energy"]
            results.append(energy)

    return np.array(results)