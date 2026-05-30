import time
import traceback
from typing import Dict, Optional

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator, BaseEstimatorV2
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import Optimizer

from src.vqe.ansatz import decompose_for_estimator


def run_vqe(
        qubit_op: SparsePauliOp,
        ansatz: QuantumCircuit,
        optimizer: Optimizer,
        estimator: Optional[BaseEstimatorV2] = None,
        initial_point: Optional[np.ndarray] = None,
        constant_energy: float = 0.0,
        seed: int = 137
) -> Dict[str, object]:
    """Run Qiskit's VQE and normalize the result into a project dictionary.

    Parameters
    ----------
    qubit_op:
        Qubit Hamiltonian represented as a sparse Pauli operator.
    ansatz:
        Parameterized variational circuit. The circuit is decomposed before
        being sent to the estimator so Aer primitives do not receive composite
        library instructions such as ``RealAmplitudes``.
    optimizer:
        Qiskit optimizer instance used by VQE.
    estimator:
        Optional Qiskit Estimator V2 primitive. When omitted, the function uses
        :class:`qiskit.primitives.StatevectorEstimator`.
    initial_point:
        Initial parameter vector. When omitted, a random vector sampled
        uniformly from ``[-pi, pi]`` is used.
    constant_energy:
        Scalar Hamiltonian energy shift added to the raw VQE eigenvalue.
    seed:
        NumPy random seed used to generate the default initial point.

    Returns
    -------
    Dict[str, object]
        On success, contains ``energy``, ``vqe_only_energy``, ``history``,
        ``eval_count``, ``total_time``, ``optimal_params``, and ``success``.
        On failure, contains ``success=False`` plus diagnostic error fields.
    """

    np.random.seed(seed)

    if estimator is None:
        estimator = StatevectorEstimator()

    history = []

    def callback(
            eval_count: int,
            params: np.ndarray,
            mean: float,
            metadata: dict[str, object],
    ) -> None:
        """Collect the optimizer energy trace reported by Qiskit VQE."""
        history.append(mean)

    try:
        estimator_ansatz = decompose_for_estimator(ansatz)

        if initial_point is None:
            initial_point = np.random.uniform(-np.pi, np.pi, estimator_ansatz.num_parameters)

        start_time = time.perf_counter()

        vqe = VQE(
            estimator=estimator,
            ansatz=estimator_ansatz,
            optimizer=optimizer,
            callback=callback,
            initial_point=initial_point
        )

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
        cause = getattr(e, "__cause__", None)
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "error_cause": str(cause) if cause is not None else None,
            "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
        }
