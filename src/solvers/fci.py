from typing import Tuple, Optional

import numpy as np
from pyscf import gto, scf, fci, mcscf


def compute_fci_energy(
        atom_string: str,
        basis: str = "sto-3g",
        active_space: Optional[Tuple[int, int]] = None,
        homo_lumo_window: int = 2,
        freeze_core: int = 0
) -> float:
    """
    Compute reference energy using FCI or CASCI with automatic active space selection.

    Parameters
    ----------
    atom_string : str
        Molecular geometry (PySCF format)
    basis : str
        Basis set
    active_space : (n_orbitals, n_electrons) or None
        Manual CAS definition
    homo_lumo_window : int
        Number of orbitals around HOMO/LUMO if CAS is automatic
    freeze_core : bool
        Whether to freeze core orbitals

    Returns
    -------
    float
        Energy in Hartree
    """

    # --- Build molecule ---
    molecule = gto.Mole()
    molecule.atom = atom_string
    molecule.basis = basis
    molecule.unit = "Angstrom"
    molecule.verbose = 0
    molecule.build()

    # --- Mean-field (Hartree-Fock) ---
    mean_field = scf.UHF(molecule)
    mean_field.kernel()

    total_electrons = molecule.nelectron
    n_occupied_orbitals = total_electrons // 2
    n_orbitals = molecule.nao_nr()

    method = None
    active_space_used = None
    source = None

    # --- CASE 1: full FCI (small systems only) ---
    if active_space is None and n_orbitals <= 10:
        ci_solver = fci.FCI(mean_field)
        energy, _ = ci_solver.kernel()

        method = "FCI"
        active_space_used = None
        source = None

        return energy

    # --- CASE 2: define active space ---
    if active_space is not None:
        n_active_orbitals, n_active_electrons = active_space

        method = "CASCI"
        active_space_used = active_space
        source = "manual"

    else:
        # Automatic HOMO-LUMO selection
        mo_energies = mean_field.mo_energy[0]

        sorted_indices = np.argsort(mo_energies)

        start = max(0, n_occupied_orbitals - homo_lumo_window)
        end = n_occupied_orbitals + homo_lumo_window

        active_indices = sorted_indices[start:end]

        n_active_orbitals = len(active_indices)
        n_active_electrons = min(total_electrons, n_active_orbitals)

        method = "CASCI"
        active_space_used = (n_active_orbitals, n_active_electrons)
        source = "auto"

    # --- CASCI ---
    active_space_solver = mcscf.CASCI(
        mean_field,
        n_active_orbitals,
        n_active_electrons,
    )

    if freeze_core:
        active_space_solver.frozen = freeze_core

    energy = active_space_solver.kernel()[0]

    return energy
