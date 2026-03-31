from typing import Dict, List

from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import QuantumCircuit

from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import Optimizer


def run_vqe(
        qubit_op: SparsePauliOp,
        ansatz: QuantumCircuit,
        optimizer: Optimizer
) -> Dict[str, object]:
    """
    Executa o algoritmo Variational Quantum Eigensolver (VQE)
    para estimar a energia fundamental do Hamiltoniano.

    Durante a execução, a energia obtida em cada iteração
    é armazenada para posterior análise de convergência.

    Parameters
    ----------
    qubit_op : SparsePauliOp
        Hamiltoniano do sistema mapeado em operadores de qubits.
    ansatz : QuantumCircuit
        Circuito variacional parametrizado.
    optimizer : Optimizer
        Otimizador clássico responsável pela minimização da energia.

    Returns
    -------
    Dict[str, object]
        Dicionário contendo:
        - energy : float
        - history : List[float]
        - iterations : int
    """

    estimator = StatevectorEstimator()

    energy_history: List[float] = []

    def callback(eval_count, params, mean, std):
        energy_history.append(mean)

    vqe = VQE(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer,
        callback=callback,
    )

    result = vqe.compute_minimum_eigenvalue(qubit_op)

    return {
        "energy": result.eigenvalue.real,
        "history": energy_history,
        "iterations": len(energy_history),
    }
