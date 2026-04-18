import time
from typing import Dict, Optional

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator, BaseEstimatorV2
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import Optimizer


def run_vqe(
        qubit_op: SparsePauliOp,
        ansatz: QuantumCircuit,
        optimizer: Optimizer,
        estimator: Optional[BaseEstimatorV2] = None,
        initial_point: Optional[np.ndarray] = None,
        constant_energy: float = 0.0,
        seed: int = 137
) -> Dict[str, object]:

    np.random.seed(seed)

    if estimator is None:
        estimator = StatevectorEstimator()

    history = []

    def callback(eval_count, params, mean, metadata):
        history.append(mean)

    if initial_point is None:
        initial_point = np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)

    start_time = time.perf_counter()

    vqe = VQE(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer,
        callback=callback,
        initial_point=initial_point
    )

    try:
        result = vqe.compute_minimum_eigenvalue(qubit_op)

        end_time = time.perf_counter()

        total_energy = result.eigenvalue.real + constant_energy

        return {
            "energy": total_energy,
            "vqe_only_energy": result.eigenvalue.real,
            "history": history,
            "eval_count": result.cost_function_evals,
            "total_time": end_time - start_time,
            "optimal_params": result.optimal_point,
            "success": True
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
