from typing import Dict, List, Optional

import numpy as np
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import QuantumCircuit

from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import Optimizer


def run_vqe(
        qubit_op: SparsePauliOp,
        ansatz: QuantumCircuit,
        optimizer: Optimizer,
        initial_point: Optional[np.ndarray] = None,
) -> Dict[str, object]:

    estimator = StatevectorEstimator()

    history: List[dict] = []

    def callback(eval_count, params, mean, std):
        history.append({
            "eval_count": eval_count,
            "energy": mean,
            "std": std,
            "params": params.copy(),
        })

    if initial_point is None:
        initial_point = np.zeros(ansatz.num_parameters, dtype=float)

    vqe = VQE(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer,
        callback=callback,
        initial_point=initial_point
    )

    result = vqe.compute_minimum_eigenvalue(qubit_op)

    return {
        "energy": result.eigenvalue.real,
        "history": history,
        "iterations": len(history),
        "optimal_params": result.optimal_point
    }
