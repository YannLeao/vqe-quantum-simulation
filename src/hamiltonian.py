from typing import Dict, Any

from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit.quantum_info import SparsePauliOp

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper


def build_qubit_hamiltonian(
        atom_string: str,
        basis: str = "sto-3g"
) -> Dict[str, Any]:
    """
    Constrói o Hamiltoniano quântico em representação de qubits
    a partir da descrição molecular fornecida.

    O Hamiltoniano eletrônico é obtido via PySCF e posteriormente
    mapeado para operadores de qubits utilizando o mapeamento
    de Jordan-Wigner. Também é calculada a energia exata (FCI)
    para fins de comparação com o VQE.

    Parameters
    ----------
    atom_string : str
        Geometria molecular no formato aceito pelo PySCFDriver.
    basis : str, optional
        Conjunto de base utilizado no cálculo eletrônico.

    Returns
    -------
    Dict[str, Any]
        Dicionário contendo:
        - qubit_op : SparsePauliOp
        - num_particles : tuple[int, int]
        - num_spatial_orbitals : int
        - fci_energy : float
        - num_qubits : int
    """

    driver = PySCFDriver(atom=atom_string, basis=basis)
    problem = driver.run()

    second_q_op = problem.hamiltonian.second_q_op()

    mapper = JordanWignerMapper()
    qubit_op: SparsePauliOp = mapper.map(second_q_op)

    # Energia exata (Full Configuration Interaction)
    exact_solver = NumPyMinimumEigensolver()
    exact_result = exact_solver.compute_minimum_eigenvalue(qubit_op)

    fci_energy = exact_result.eigenvalue.real

    return {
        "qubit_op": qubit_op,
        "num_particles": problem.num_particles,
        "num_spatial_orbitals": problem.num_spatial_orbitals,
        "fci_energy": fci_energy,
        "num_qubits": qubit_op.num_qubits,
    }
