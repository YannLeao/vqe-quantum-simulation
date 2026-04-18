import csv
import json
import time
from pathlib import Path
from typing import Optional, Tuple, Callable

import numpy as np
from pyscf import gto

from src.solvers.fci import compute_fci_energy
from src.utils.paths import get_project_root


def cache_fci(
        molecule: str,
        geometry_fn: Callable[[float], str],
        distances: np.ndarray,
        basis: str = "sto-3g",
        active_space: Optional[Tuple[int, int]] = None,
        homo_lumo_window: int = 2,
        freeze_core: int = 0,
        data_dir: Optional[Path] = None,
        overwrite: bool = True,
        verbose: bool = True,
):
    # --- Setup paths ---
    if data_dir is None:
        data_dir = get_project_root() / "data"

    path = data_dir / molecule / basis
    path.mkdir(parents=True, exist_ok=True)

    # -- Experiment signature
    config = {
        "molecule": molecule,
        "basis": basis,
        "active_space": active_space,
        "homo_lumo_window": homo_lumo_window,
        "freeze_core": freeze_core
    }

    config_str = json.dumps(config, sort_keys=True)
    config_hash = str(abs(hash(config_str)))[:10]

    file_path = path / f"fci_{config_hash}.csv"
    meta_path = path / f"fci_{config_hash}.json"

    # --- Load existing cache ---
    done = {}

    if file_path.exists() and not overwrite:
        with open(file_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                d = float(row["distance"])
                e = float(row["fci_energy"])
                done[d] = e

    # --- Save metadata (once) ---
    if not meta_path.exists() or overwrite:
        with open(meta_path, "w") as f:
            json.dump(config, f, indent=4)
    
    # --- Prepare file ---
    write_header = not file_path.exists() or overwrite

    f = open(file_path, "a", newline="")
    writer = csv.writer(f)

    if write_header:
        writer.writerow([
            "distance",
            "energy",
            "method",
            "n_orbitals",
            "n_electrons",
            "timestamp"
        ])
        f.flush()

    # --- Main loop ---
    results = []
    start_time = time.time()

    try:
        for i, d in enumerate(distances):

            # --- Cache Lookup ---
            cached_value = None
            for x in done:
                if abs(d - x) < 1e-8:
                    cached_value = done[x]
                    break

            if cached_value is not None:
                results.append(cached_value)
                if verbose:
                    print(f"[{i+1}/{len(distances)}] d={d:.4f} (cached)")
                continue

            # --- Compute ---
            try:
                atom_string = geometry_fn(d)

                #
                mol = gto.Mole()
                mol.atom = atom_string
                mol.basis = basis
                mol.unit = "Angstrom"
                mol.build()

                n_orbitals = mol.nao_nr()
                n_electrons = mol.nelectron

                energy = compute_fci_energy(
                    atom_string=atom_string,
                    basis=basis,
                    active_space=active_space,
                    homo_lumo_window=homo_lumo_window,
                    freeze_core=freeze_core
                )

                if energy is None:
                    raise ValueError("FCI energy returned None")

                results.append(energy)
                done[d] = energy

                # --- Detect method ---
                method = "FCI" if active_space is None and n_orbitals <= 10 else "CASCI"

                # --- Save ---
                writer.writerow([
                    d,
                    energy,
                    method,
                    n_orbitals,
                    n_electrons,
                    time.time()
                ])
                f.flush()

                if verbose:
                    elapsed = time.time() - start_time
                    print(
                        f"[{i+1}/{len(distances)}] d={d:.4f} "
                        f"E={energy:.8f} ({method}) "
                        f"orb={n_orbitals} ⏱ {elapsed:.1f}s"
                    )
            except Exception:
                results.append(np.nan)

    finally:
        f.close()

    return np.array(results)
