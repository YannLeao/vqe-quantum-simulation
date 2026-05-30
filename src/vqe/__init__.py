__all__ = [
    "MolecularSystem",
    "build_electronic_problem",
    "build_electronic_hamiltonian",
    "build_qubit_hamiltonian",
    "default_statevector_systems",
    "IBM_HERON_R2_BACKENDS",
    "pauli_terms_from_qubit_hamiltonian",
    "run_vqe_grid_search",
    "statevector_grid_systems",
    "SyntheticNoiseConfig",
    "build_synthetic_noisy_aer_estimator",
]


def __getattr__(name: str):
    """Lazily expose VQE helpers while keeping optional imports lightweight."""
    if name in {
        "MolecularSystem",
        "default_statevector_systems",
        "statevector_grid_systems",
    }:
        from src.vqe.molecular_system import (
            MolecularSystem,
            default_statevector_systems,
            statevector_grid_systems,
        )

        return {
            "MolecularSystem": MolecularSystem,
            "default_statevector_systems": default_statevector_systems,
            "statevector_grid_systems": statevector_grid_systems,
        }[name]

    if name == "run_vqe_grid_search":
        from src.vqe.grid_search import run_vqe_grid_search

        return run_vqe_grid_search

    if name in {
        "IBM_HERON_R2_BACKENDS",
        "SyntheticNoiseConfig",
        "build_synthetic_noisy_aer_estimator",
    }:
        from src.vqe.noise import (
            IBM_HERON_R2_BACKENDS,
            SyntheticNoiseConfig,
            build_synthetic_noisy_aer_estimator,
        )

        return {
            "IBM_HERON_R2_BACKENDS": IBM_HERON_R2_BACKENDS,
            "SyntheticNoiseConfig": SyntheticNoiseConfig,
            "build_synthetic_noisy_aer_estimator": build_synthetic_noisy_aer_estimator,
        }[name]

    if name in {
        "build_electronic_problem",
        "build_electronic_hamiltonian",
        "build_qubit_hamiltonian",
        "pauli_terms_from_qubit_hamiltonian",
    }:
        from src.vqe.hamiltonian import (
            build_electronic_problem,
            build_electronic_hamiltonian,
            build_qubit_hamiltonian,
            pauli_terms_from_qubit_hamiltonian,
        )

        return {
            "build_electronic_problem": build_electronic_problem,
            "build_electronic_hamiltonian": build_electronic_hamiltonian,
            "build_qubit_hamiltonian": build_qubit_hamiltonian,
            "pauli_terms_from_qubit_hamiltonian": pauli_terms_from_qubit_hamiltonian,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
