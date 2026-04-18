from typing import Optional

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2, RealAmplitudes, ExcitationPreserving

from qiskit_nature.second_q.circuit.library import UCCSD, PUCCSD
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
    mapper = JordanWignerMapper()

    #  --- Physically-Inspired ---
    if name in ["uccsd", "puccsd"]:
        if num_particles is None or num_spatial_orbitals is None:
            raise ValueError(f"The {name} ansatz requires num_particles and num_spatial_orbitals")

        initial_state = HartreeFock(
            num_spatial_orbitals=num_spatial_orbitals,
            num_particles=num_particles,
            qubit_mapper=mapper,
        )

        if name == "uccsd":
            return UCCSD(
                num_spatial_orbitals=num_spatial_orbitals,
                num_particles=num_particles,
                qubit_mapper=mapper,
                initial_state=initial_state,
            )
        elif name == "puccsd":
            return PUCCSD(
                num_spatial_orbitals=num_spatial_orbitals,
                num_particles=num_particles,
                qubit_mapper=mapper,
                initial_state=initial_state,
            )

    # --- Hardware-efficient ---
    elif name == "efficient_su2":
        return EfficientSU2(num_qubits, reps=reps, entanglement="linear")
    elif name == "real_amplitudes":
        return RealAmplitudes(num_qubits, reps=reps, entanglement="linear")
    elif name == "excitation_preserving":
        return ExcitationPreserving(num_qubits, reps=reps, entanglement="linear")


    else:
        raise ValueError(f"Unknown ansatz: {name}")
