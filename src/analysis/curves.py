import numpy as np

from src.pipeline import run_experiment
from src.vqe.hamiltonian import build_qubit_hamiltonian


def compute_vqe_curve(
        config: dict,
        distances: np.ndarray,
        initial_point=None
):
    energies = []

    for d in distances:
        config["geometry"] = f"H 0 0 0; H 0 0 {d}"

        result = run_experiment(
            config,
            initial_point=initial_point
        )

        initial_point = result["optimal_params"]

        energies.append(result["energy"])

    return np.array(energies)


def compute_fci_curve(distances: np.ndarray, basis="sto-3g"):
    results = []

    for d in distances:
        atom_string = f"H 0 0 0; H 0 0 {d}"

        data = build_qubit_hamiltonian(
            atom_string=atom_string,
            basis=basis
        )

        energy = data["fci_energy"]
        results.append(energy)

    return np.array(results)
