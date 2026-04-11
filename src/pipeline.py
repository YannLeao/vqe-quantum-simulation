import time

from src.vqe.ansatz import build_ansatz
from src.vqe.hamiltonian import build_qubit_hamiltonian
from src.vqe.optimizer import get_optimizer
from src.vqe.vqe_runner import run_vqe


def run_experiment(config: dict, initial_point=None):

    timings = {}

    # Tempo total
    t_total_start = time.perf_counter()

    t0 = time.perf_counter()
    data = build_qubit_hamiltonian(
        atom_string=config["geometry"],
        basis=config["basis"]
    )
    timings["hamiltonian"] = time.perf_counter() - t0

    ansatz = build_ansatz(
        name=config["ansatz"],
        num_qubits=data["num_qubits"],
        reps=config.get("reps", 2),
        num_particles=data["num_particles"],
        num_spatial_orbitals=data["num_spatial_orbitals"],
    )

    optimizer = get_optimizer(
        name=config["optimizer"],
        max_iter=config["max_iter"]
    )

    t0 = time.perf_counter()
    result = run_vqe(
        data["qubit_op"],
        ansatz,
        optimizer,
        initial_point=initial_point
    )
    timings["vqe"] = time.perf_counter() - t0
    timings["total"] = time.perf_counter() - t_total_start

    total_energy = result["energy"] + data["nuclear_energy"]

    return {
        "energy": total_energy,
        "fci": data["fci_energy"],
        "error": abs(total_energy - data["fci_energy"]),
        "iterations": result["iterations"],
        "optimal_params": result["optimal_params"],
        "config": config.copy(),
        "timings": timings,
        "ansatz_circuit": ansatz
    }
