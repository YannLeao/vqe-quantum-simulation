from typing import Dict, Optional, Sequence, Tuple, cast

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import BravyiKitaevMapper, JordanWignerMapper, ParityMapper
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.transformers import FreezeCoreTransformer, ActiveSpaceTransformer

# Compatibility for dependencies that still call the removed NumPy alias.
if not hasattr(np, "in1d"):
    np.in1d = np.isin


def build_electronic_hamiltonian(
        atom_string: str,
        basis: str = "sto-3g",
        active_space: Optional[Tuple[int, int]] = None,
        active_orbitals: Optional[Sequence[int]] = None,
        homo_lumo_window: int = 2,
        freeze_core: bool = True
) -> Tuple[FermionicOp, float]:
    """Build a second-quantized electronic Hamiltonian and scalar shift.

    Parameters
    ----------
    atom_string:
        PySCF atom specification, for example ``"H 0 0 0; H 0 0 0.735"``.
    basis:
        Atomic basis set passed to ``PySCFDriver``.
    active_space:
        Optional active space ``(n_electrons, n_spatial_orbitals)``. When
        provided, it takes precedence over ``homo_lumo_window``.
    active_orbitals:
        Optional explicit spatial orbital indices selected by
        ``ActiveSpaceTransformer``.
    homo_lumo_window:
        Number of occupied/virtual orbital pairs used to construct a fallback
        active space when ``active_space`` is not provided. ``0`` keeps the full
        transformed problem.
    freeze_core:
        Whether to apply ``FreezeCoreTransformer`` before active-space
        selection.

    Returns
    -------
    Tuple[FermionicOp, float]
        Fermionic Hamiltonian and the scalar energy offset containing nuclear
        repulsion and transformer shifts.
    """

    try:
        # Some qiskit_nature versions expose this keyword.
        driver = PySCFDriver(atom=atom_string, basis=basis, initial_guess="hcore")
    except TypeError:
        # Fallback for versions that do not support initial_guess.
        driver = PySCFDriver(atom=atom_string, basis=basis)
    problem = driver.run()


    if freeze_core:
        core_transformer = FreezeCoreTransformer()
        problem = core_transformer.transform(problem)

    # Full-space mode: represent full problem explicitly as an active space equal
    # to the full system, which keeps metadata consistent for tapering workflows.
    if active_space is not None:
        n_active_electrons, n_active_orbitals = active_space
    elif homo_lumo_window > 0:
        n_active_orbitals = homo_lumo_window * 2
        n_active_electrons = n_active_orbitals
    else:
        n_alpha, n_beta = problem.num_particles
        n_active_electrons = int(n_alpha + n_beta)
        n_active_orbitals = int(problem.num_spatial_orbitals)

    transformer = ActiveSpaceTransformer(
        num_electrons=n_active_electrons,
        num_spatial_orbitals=n_active_orbitals,
        active_orbitals=list(active_orbitals) if active_orbitals is not None else None,
    )
    problem = cast(ElectronicStructureProblem, transformer.transform(problem))

    fermionic_op = problem.hamiltonian.second_q_op()

    # Sum all scalar offsets tracked by Qiskit Nature (nuclear repulsion,
    # freeze-core shifts, active-space shifts, etc.) to keep a consistent
    # energy zero between qubit and molecular references.
    constant_energy = float(
        sum(float(np.real(v)) for v in problem.hamiltonian.constants.values())
    )

    return fermionic_op, constant_energy


def build_electronic_problem(
        atom_string: str,
        basis: str = "sto-3g",
        active_space: Optional[Tuple[int, int]] = None,
    active_orbitals: Optional[Sequence[int]] = None,
        homo_lumo_window: int = 2,
        freeze_core: bool = True,
) -> ElectronicStructureProblem:
    """Build and transform a Qiskit Nature electronic-structure problem.

    This function is the canonical entry point for molecular problem setup in
    the project. It runs PySCF, optionally freezes core orbitals, and then
    applies an active-space transformation so downstream VQE/FCI code sees a
    consistent reduced problem.

    Parameters
    ----------
    atom_string:
        PySCF atom specification.
    basis:
        Atomic basis set passed to ``PySCFDriver``.
    active_space:
        Optional active space ``(n_electrons, n_spatial_orbitals)``.
    active_orbitals:
        Optional explicit spatial orbital indices.
    homo_lumo_window:
        Fallback active-space window used only when ``active_space`` is not
        provided.
    freeze_core:
        Whether to apply ``FreezeCoreTransformer`` before active-space
        selection.

    Returns
    -------
    ElectronicStructureProblem
        Transformed electronic problem with Hamiltonian constants preserved.
    """

    try:
        driver = PySCFDriver(atom=atom_string, basis=basis, initial_guess="hcore")
    except TypeError:
        driver = PySCFDriver(atom=atom_string, basis=basis)
    problem = driver.run()

    if freeze_core:
        core_transformer = FreezeCoreTransformer()
        problem = core_transformer.transform(problem)

    if active_space is not None:
        n_active_electrons, n_active_orbitals = active_space
    elif homo_lumo_window > 0:
        n_active_orbitals = homo_lumo_window * 2
        n_active_electrons = n_active_orbitals
    else:
        n_alpha, n_beta = problem.num_particles
        n_active_electrons = int(n_alpha + n_beta)
        n_active_orbitals = int(problem.num_spatial_orbitals)

    transformer = ActiveSpaceTransformer(
        num_electrons=n_active_electrons,
        num_spatial_orbitals=n_active_orbitals,
        active_orbitals=list(active_orbitals) if active_orbitals is not None else None,
    )
    problem = cast(ElectronicStructureProblem, transformer.transform(problem))

    return problem

def build_qubit_hamiltonian(
        electronic_hamiltonian: FermionicOp,
        mapper: str = "jw",
        z2symmetry_reduction: bool = False,
        problem: Optional[ElectronicStructureProblem] = None,
        num_particles: Optional[Tuple[int, int]] = None,
) -> SparsePauliOp:
    """Map a fermionic Hamiltonian to a qubit Hamiltonian.

    Parameters
    ----------
    electronic_hamiltonian:
        Fermionic Hamiltonian returned by Qiskit Nature.
    mapper:
        Mapping strategy. Supported values are ``"jw"`` for Jordan-Wigner,
        ``"bk"`` for Bravyi-Kitaev, and ``"parity"`` for parity mapping.
    z2symmetry_reduction:
        Whether to use Qiskit Nature's tapered mapper. This requires
        ``problem`` because the symmetry sector is inferred from the electronic
        structure problem.
    problem:
        Electronic problem used to build a tapered mapper when
        ``z2symmetry_reduction=True``.
    num_particles:
        Number of alpha/beta particles used by the parity mapper.

    Returns
    -------
    SparsePauliOp
        Qubit Hamiltonian ready for VQE, FCI reference diagonalization, or
        Fourier energy sampling.
    """

    if mapper == "jw":
        mapper_obj = JordanWignerMapper()
    elif mapper == "bk":
        mapper_obj = BravyiKitaevMapper()
    elif mapper == "parity":
        mapper_obj = ParityMapper(num_particles=num_particles)
    else:
        raise ValueError(f"Mapper {mapper} not supported")

    if z2symmetry_reduction:
        if problem is None:
            raise ValueError("problem is required when z2symmetry_reduction=True")
        tapered_mapper = problem.get_tapered_mapper(mapper_obj)
        qubit_op = tapered_mapper.map(electronic_hamiltonian)
    else:
        qubit_op = mapper_obj.map(electronic_hamiltonian)

    return qubit_op


def pauli_terms_from_qubit_hamiltonian(
        qubit_hamiltonian: SparsePauliOp,
        tolerance: float = 1e-12,
        real_only: bool = True,
) -> list[tuple[str, float | complex]]:
    """Extract Pauli-string terms from a qubit Hamiltonian.

    Parameters
    ----------
    qubit_hamiltonian : SparsePauliOp
        Hamiltonian already mapped to qubit space.
    tolerance : float
        Terms with |coef| <= tolerance are dropped.
    real_only : bool
        If True, return real coefficients when the imaginary part is negligible.

    Returns
    -------
    list[tuple[str, float | complex]]
        List of (pauli_label, coefficient), e.g. [("II", c0), ("IZ", c1), ...].
    """
    simplified = qubit_hamiltonian.simplify(atol=tolerance)

    labels = simplified.paulis.to_labels()
    coeffs = simplified.coeffs

    terms: list[tuple[str, float | complex]] = []
    for label, coeff in zip(labels, coeffs):
        if abs(coeff) <= tolerance:
            continue

        if real_only and abs(coeff.imag) <= tolerance:
            terms.append((label, float(coeff.real)))
        else:
            terms.append((label, complex(coeff)))

    return terms


def extract_problem_metadat(problem: ElectronicStructureProblem) -> Dict[str, Optional[float | int | Tuple[int, int]]]:
    """Extract compact metadata from an electronic-structure problem.

    Parameters
    ----------
    problem:
        Qiskit Nature electronic problem.

    Returns
    -------
    Dict[str, Optional[float | int | Tuple[int, int]]]
        Particle count, number of spatial orbitals, and nuclear repulsion
        energy. The function name keeps its historical spelling for backwards
        compatibility.
    """

    return {
        "num_particles": problem.num_particles,
        "num_spatial_orbitals": problem.num_spatial_orbitals,
        "nuclear_energy": problem.nuclear_repulsion_energy,
    }

