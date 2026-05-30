import time
from typing import Dict, Any, Optional

import numpy as np
from qiskit.primitives import BaseEstimatorV2

from src.vqe.ansatz import build_ansatz
from src.vqe.hamiltonian import build_qubit_hamiltonian, build_electronic_problem, electronic_constant_energy
from src.vqe.optimizer import get_optimizer
from src.vqe.vqe_runner import run_vqe


def run_experiment(
        config: Dict[str, Any],
        initial_point: Optional[np.ndarray] = None,
        estimator: Optional[BaseEstimatorV2] = None,
) -> Dict[str, Any]:
    """Run one complete molecular VQE experiment.

    The pipeline builds the electronic problem, maps it to qubits, constructs
    the requested ansatz and optimizer, executes VQE, and returns a normalized
    result dictionary with timing and Hamiltonian metadata.

    Parameters
    ----------
    config:
        Experiment configuration. Required keys are ``geometry``, ``basis``,
        ``ansatz``, and ``optimizer``. Common optional keys include
        ``active_space``, ``active_orbitals``, ``homo_lumo_window``,
        ``freeze_core``, ``mapper``, ``z2symmetry_reduction``, ``reps``,
        ``max_iter``, and ``seed``.
    initial_point:
        Optional initial parameter vector passed to VQE. When omitted, the VQE
        runner samples a random point from ``[-pi, pi]``.
    estimator:
        Optional Qiskit Estimator V2 primitive. Use ``None`` for the default
        statevector estimator, or pass an Aer estimator for noisy simulation.

    Returns
    -------
    Dict[str, Any]
        Normalized experiment result. Successful runs include ``energy``,
        ``vqe_raw_energy``, ``eval_count``, ``optimal_params``, ``history``,
        ``timings``, ``config``, and ``metadata``. Failed runs keep the same
        outer structure and include ``error``/``error_type``.
    """

    timings = {}

    t_start_total = time.perf_counter()

    # --- Build electronic problem and mapping qubit hamiltonian ---
    t0 = time.perf_counter()
    problem = build_electronic_problem(
        atom_string=config["geometry"],
        basis=config["basis"],
        active_space=config.get("active_space"),
        active_orbitals=config.get("active_orbitals"),
        homo_lumo_window=config.get("homo_lumo_window", 0),
        freeze_core=config.get("freeze_core", 0)
    )
    fermionic_op = problem.hamiltonian.second_q_op()
    constant_energy = electronic_constant_energy(problem)
    qubit_op = build_qubit_hamiltonian(
        fermionic_op,
        mapper=config.get("mapper", "jw"),
        z2symmetry_reduction=config.get("z2symmetry_reduction", False),
        problem=problem,
        num_particles=problem.num_particles,
    )
    timings["setup_hamiltonian"] = time.perf_counter() - t0

    # --- Ansatz and optimizer setup ---
    num_qubits = qubit_op.num_qubits
    ansatz = build_ansatz(
        name=config["ansatz"],
        num_qubits=num_qubits,
        reps=config.get("reps", 1),
        num_particles=config.get("num_particles", problem.num_particles),
        num_spatial_orbitals=config.get("num_spatial_orbitals", problem.num_spatial_orbitals)
    )

    optimizer = get_optimizer(
        name=config["optimizer"],
        max_iter=config.get("max_iter", 200)
    )

    # --- Run VQE ---
    vqe_result = run_vqe(
        qubit_op=qubit_op,
        ansatz=ansatz,
        optimizer=optimizer,
        estimator=estimator,
        initial_point=initial_point,
        constant_energy=constant_energy,
        seed=config.get("seed", 137),
    )
    timings["vqe_execution"] = vqe_result.get("total_time")
    timings["total_experiment"] = time.perf_counter() - t_start_total

    if not vqe_result["success"]:
        return {
            "energy": np.nan,
            "vqe_raw_energy": np.nan,
            "eval_count": 0,
            "optimal_params": None,
            "history": [],
            "success": False,
            "error": vqe_result.get("error"),
            "error_type": vqe_result.get("error_type"),
            "timings": timings,
            "config": config.copy(),
            "metadata": {
                "num_qubits": qubit_op.num_qubits,
                "num_terms": len(qubit_op.paulis),
                "constant_energy": constant_energy,
            },
        }

    return {
        "energy": vqe_result["energy"],
        "vqe_raw_energy": vqe_result["vqe_only_energy"],
        "eval_count": vqe_result["eval_count"],
        "optimal_params": vqe_result["optimal_params"],
        "history": vqe_result["history"],
        "success": vqe_result["success"],
        "timings": timings,
        "config": config.copy(),
        "metadata": {
            "num_qubits": qubit_op.num_qubits,
            "num_terms": len(qubit_op.paulis),
            "constant_energy": constant_energy,
        },
    }
