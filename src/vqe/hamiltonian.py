from typing import Dict, Any

from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit.quantum_info import SparsePauliOp

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper


def build_qubit_hamiltonian(
        atom_string: str,
        basis: str = "sto-3g",
        calculate_fci=False

) -> Dict[str, Any]:

    driver = PySCFDriver(
        atom=atom_string,
        basis=basis
    )

    problem = driver.run()

    second_q_op = problem.second_q_ops()[0]

    mapper = JordanWignerMapper()
    qubit_op: SparsePauliOp = mapper.map(second_q_op)

    nuclear_energy = problem.nuclear_repulsion_energy

    if calculate_fci:
        exact_solver = NumPyMinimumEigensolver()
        result_fci = exact_solver.compute_minimum_eigenvalue(qubit_op)
        total_fci_energy = result_fci.eigenvalue.real + nuclear_energy
    else:
        total_fci_energy = 0.0

    return {
        "qubit_op": qubit_op,
        "num_particles": problem.num_particles,
        "num_spatial_orbitals": problem.num_spatial_orbitals,
        "fci_energy": total_fci_energy if calculate_fci else None,
        "num_qubits": qubit_op.num_qubits,
        "nuclear_energy": nuclear_energy,
    }

def get_atom_energy(
        atom_symbol: str,
        basis: str = "sto-3g",
        charge: int = 0,
        spin: int = 0,
):
    """
    Calcula energia atômica via FCI.

    Parameters
    ----------
    atom_symbol : str
        Ex: "H", "Li", "O"
    spin : int
        2S = número de elétrons desemparelhados
    """

    atom_string = f"{atom_symbol} 0 0 0"

    driver = PySCFDriver(
        atom=atom_string,
        basis=basis,
        charge=charge,
        spin=spin,
    )

    problem = driver.run()

    second_q_op = problem.second_q_ops()[0]

    mapper = JordanWignerMapper()
    qubit_op = mapper.map(second_q_op)

    nuclear_energy = problem.nuclear_repulsion_energy

    exact_solver = NumPyMinimumEigensolver()
    result = exact_solver.compute_minimum_eigenvalue(qubit_op)

    total_energy = result.eigenvalue.real + nuclear_energy

    return total_energy