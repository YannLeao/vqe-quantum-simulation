from pyscf import gto, scf, fci


def compute_fci_energy(atom_string: str, basis: str = "sto-3g") -> float:
    """
    Compute the Full Configuration Interaction (FCI) energy using PySCF.

    Parameters
    ----------
    atom_string : str
        Molecular geometry in PySCF format (e.g., "Li 0 0 0; H 0 0 1.6").
    basis : str, optional
        Basis set to use, by default "sto-3g".

    Returns
    -------
    float
        Ground state electronic energy from FCI (in Hartree).

    Notes
    -----
    This function uses PySCF's direct FCI solver, avoiding qubit mapping
    and significantly reducing computational overhead compared to
    qubit-based exact diagonalization methods.
    """

    mol = gto.M(
        atom=atom_string,
        basis=basis,
        unit="Angstrom"
    )

    mf = scf.RHF(mol)
    mf.kernel()

    cisolver = fci.FCI(mol, mf.mo_coeff)
    energy, _ = cisolver.kernel()

    return energy
