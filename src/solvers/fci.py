from pyscf import gto, scf, fci


def compute_fci_energy(atom_string: str, basis: str = "sto-3g") -> float:
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
