from typing import Dict, Optional, Tuple, cast

from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.drivers import PySCFDriver, InitialGuess
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.transformers import FreezeCoreTransformer, ActiveSpaceTransformer


def build_electronic_hamiltonian(
        atom_string: str,
        basis: str = "sto-3g",
        active_space: Optional[Tuple[int, int]] = None,
        homo_lumo_window: int = 2,
        freeze_core: bool = True
) -> Tuple[FermionicOp, float]:

    driver = PySCFDriver(atom=atom_string, basis=basis, initial_guess=InitialGuess.HCORE)
    problem = driver.run()


    # --- Define active space ---
    if active_space:
        n_active_electrons, n_active_orbitals = active_space
    else:
        n_active_orbitals = homo_lumo_window * 2
        n_active_electrons = n_active_orbitals

    transformer = ActiveSpaceTransformer(
        num_electrons=n_active_electrons,
        num_spatial_orbitals=n_active_orbitals,
        active_orbitals=None
    )

    if freeze_core:
        core_transformer = FreezeCoreTransformer()
        problem = core_transformer.transform(problem)

    problem = cast(ElectronicStructureProblem, transformer.transform(problem))

    fermionic_op = problem.hamiltonian.second_q_op()

    constant_energy = problem.nuclear_repulsion_energy + problem.hamiltonian.constants.get('ElectronicEnergy', 0)

    return fermionic_op, constant_energy

def build_qubit_hamiltonian(
        electronic_hamiltonian: FermionicOp,
        mapper: str = "jw"
) -> SparsePauliOp:

    if mapper == "jw":
        mapper_obj = JordanWignerMapper()
    else:
        raise ValueError(f"Mapper {mapper} not supported")

    qubit_op = mapper_obj.map(electronic_hamiltonian)

    return qubit_op


def extract_problem_metadat(problem: ElectronicStructureProblem) -> Dict[str, Optional[float | int | Tuple[int, int]]]:

    return {
        "num_particles": problem.num_particles,
        "num_spatial_orbitals": problem.num_spatial_orbitals,
        "nuclear_energy": problem.nuclear_repulsion_energy,
    }

