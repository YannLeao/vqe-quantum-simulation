import time
from typing import Dict, Any, Optional

import numpy as np

from src.vqe.ansatz import build_ansatz
from src.vqe.hamiltonian import build_qubit_hamiltonian, build_electronic_hamiltonian
from src.vqe.optimizer import get_optimizer
from src.vqe.vqe_runner import run_vqe


def run_experiment(
        config: Dict[str, Any],
        initial_point: Optional[np.ndarray] = None
) -> Dict[str, Any]:

    timings = {}

    t_start_total = time.perf_counter()

    # --- Build electronic problem and mapping qubit hamiltonian ---
    t0 = time.perf_counter()
    fermionic_op, constant_energy = build_electronic_hamiltonian(
        atom_string=config["geometry"],
        basis=config["basis"],
        active_space=config.get("active_space"),
        homo_lumo_window=config.get("homo_lumo_window", 0),
        freeze_core=config.get("freeze_core", 0)
    )
    qubit_op = build_qubit_hamiltonian(fermionic_op, mapper="jw")
    timings["setup_hamiltonian"] = time.perf_counter() - t0

    # --- Ansatz and optimizer setup ---
    num_qubits = qubit_op.num_qubits
    ansatz = build_ansatz(
        name=config["ansatz"],
        num_qubits=num_qubits,
        reps=config.get("reps", 1),
        num_particles=config.get("num_particles"),
        num_spatial_orbitals=config.get("num_spatial_orbitals")
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
        initial_point=initial_point,
        constant_energy=constant_energy
    )
    timings["vqe_execution"] = vqe_result["total_time"]
    timings["total_experiment"] = time.perf_counter() - t_start_total

    return {
        "energy": vqe_result["energy"],
        "vqe_raw_energy": vqe_result["vqe_only_energy"],
        "eval_count": vqe_result["eval_count"],
        "optimal_params": vqe_result["optimal_params"],
        "success": vqe_result["success"],
        "timings": timings,
        "config": config.copy()
    }
