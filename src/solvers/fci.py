from typing import Optional, Sequence, Tuple

import numpy as np
from pyscf import gto, scf, fci, mcscf


def compute_fci_energy(
        atom_string: str,
        basis: str = "sto-3g",
        active_space: Optional[Tuple[int, int]] = None,
    active_orbitals: Optional[Sequence[int]] = None,
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
    active_space : (n_electrons, n_orbitals) or None
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
    mean_field = scf.RHF(molecule)
    mean_field.kernel()

    total_electrons = molecule.nelectron
    n_occupied_orbitals = total_electrons // 2
    n_orbitals = molecule.nao_nr()

    # --- CASE 1: full FCI (small systems only) ---
    if active_space is None and n_orbitals <= 10:
        ci_solver = fci.FCI(mean_field)
        energy, _ = ci_solver.kernel()
        return energy

    # --- CASE 2: define active space ---
    if active_space is not None:
        n_active_electrons, n_active_orbitals = active_space

    else:
        # Automatic HOMO-LUMO selection
        mo_energies = mean_field.mo_energy

        sorted_indices = np.argsort(mo_energies)

        start = max(0, n_occupied_orbitals - homo_lumo_window)
        end = min(len(sorted_indices), n_occupied_orbitals + homo_lumo_window)

        active_indices = sorted_indices[start:end]

        n_active_orbitals = len(active_indices)

        # Keep frozen-core electrons out of active space and enforce valid parity.
        n_active_electrons = max(0, total_electrons - 2 * freeze_core)
        n_active_electrons = min(n_active_electrons, 2 * n_active_orbitals)
        if n_active_electrons % 2 != 0:
            n_active_electrons -= 1
        if n_active_electrons <= 0 or n_active_orbitals <= 0:
            raise ValueError("Invalid automatic active space selection.")

    # --- CASCI ---
    active_space_solver = mcscf.CASCI(
        mean_field,
        n_active_orbitals,
        n_active_electrons,
    )

    if freeze_core:
        active_space_solver.frozen = freeze_core

    if active_orbitals is not None:
        if len(active_orbitals) != n_active_orbitals:
            raise ValueError(
                "active_orbitals length must match active_space num_orbitals"
            )

        # PySCF CASCI expects MO indices in the full-orbital basis.
        # Our pipeline uses indices after frozen cores are removed, so we shift
        # by the number of frozen orbitals to align conventions.
        cas_mo_list = [int(idx) + int(freeze_core) for idx in active_orbitals]
        mo_sorted = active_space_solver.sort_mo(cas_mo_list)
        energy = active_space_solver.kernel(mo_coeff=mo_sorted)[0]
    else:
        energy = active_space_solver.kernel()[0]

    return energy
