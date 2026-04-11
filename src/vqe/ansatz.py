from typing import Optional

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2

from qiskit_nature.second_q.circuit.library import UCCSD
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock


def build_ansatz(
        name: str,
        num_qubits: int,
        reps: int = 1,
        num_particles: Optional[tuple[int, int]] = None,
        num_spatial_orbitals: Optional[int] = None,
) -> QuantumCircuit:

    name = name.lower()

    # 1. Hardware-efficient
    if name == "efficient_su2":
        return EfficientSU2(
            num_qubits=num_qubits,
            reps=reps,
            entanglement="linear"
        )
    # 2. Physically-Inspired Ansatz
    elif name == "uccsd":

        if num_particles is None or num_spatial_orbitals is None:
            raise ValueError("UCCSD requires num_particles and num_spatial_orbitals")

        mapper = JordanWignerMapper()

        initial_state = HartreeFock(
            num_spatial_orbitals=num_spatial_orbitals,
            num_particles=num_particles,
            qubit_mapper=mapper,
        )

        ansatz = UCCSD(
            num_spatial_orbitals=num_spatial_orbitals,
            num_particles=num_particles,
            qubit_mapper=mapper,
            initial_state=initial_state,
        )

        return ansatz
    else:
        raise ValueError(f"Unknown ansatz: {name}")