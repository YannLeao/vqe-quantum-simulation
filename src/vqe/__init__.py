__all__ = [
    "MolecularSystem",
    "build_electronic_problem",
    "build_electronic_hamiltonian",
    "build_qubit_hamiltonian",
    "default_statevector_systems",
    "pauli_terms_from_qubit_hamiltonian",
    "run_vqe_grid_search",
    "statevector_grid_systems",
]


def __getattr__(name: str):
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
