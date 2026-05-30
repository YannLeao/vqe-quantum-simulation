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
    """Build a VQE ansatz circuit from a project-level name.

    Parameters
    ----------
    name:
        Ansatz identifier. Supported values are ``"real_amplitudes"``,
        ``"efficient_su2"``, ``"excitation_preserving"``, ``"uccsd"``, and
        ``"puccsd"``.
    num_qubits:
        Number of qubits in the target qubit Hamiltonian. This is required for
        hardware-efficient ansatz families.
    reps:
        Repetition depth for hardware-efficient ansatz families. Qiskit
        Nature ansatz classes such as UCCSD do not use this value.
    num_particles:
        Number of alpha and beta electrons, ``(n_alpha, n_beta)``. Required by
        physically inspired ansatz families.
    num_spatial_orbitals:
        Number of spatial orbitals in the active electronic problem. Required
        by physically inspired ansatz families.

    Returns
    -------
    QuantumCircuit
        Parameterized ansatz circuit ready to be passed to VQE.

    Raises
    ------
    ValueError
        If ``name`` is unknown or if a physically inspired ansatz is requested
        without ``num_particles`` and ``num_spatial_orbitals``.
    """

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
        else:
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


def decompose_for_estimator(ansatz: QuantumCircuit) -> QuantumCircuit:
    """Expand composite ansatz instructions before estimator execution.

    Some Qiskit primitives, especially Aer with an explicit noise model, expect
    circuits expressed in simulator-supported basis instructions. Library
    circuits such as ``RealAmplitudes`` may otherwise appear as a single custom
    instruction and fail with errors like ``unknown instruction``.

    Parameters
    ----------
    ansatz:
        Parameterized circuit produced by :func:`build_ansatz` or an equivalent
        Qiskit constructor.

    Returns
    -------
    QuantumCircuit
        Decomposed circuit when decomposition preserves the parameter count;
        otherwise the original circuit is returned.
    """
    decomposed = ansatz.decompose(reps=10)
    return decomposed if decomposed.num_parameters == ansatz.num_parameters else ansatz
