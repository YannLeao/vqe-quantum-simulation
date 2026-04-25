import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from pyscf import gto

from src.solvers.fci import compute_fci_energy
from src.utils.paths import get_project_root


def _build_cache_config(
        molecule: str,
        basis: str,
        active_space: Optional[Tuple[int, int]],
    active_orbitals: Optional[Tuple[int, ...]],
        homo_lumo_window: int,
        freeze_core: int,
) -> dict:
    return {
        "molecule": molecule,
        "basis": basis,
        "active_space": active_space,
    "active_orbitals": active_orbitals,
        "homo_lumo_window": homo_lumo_window,
        "freeze_core": freeze_core,
    }


def _stable_config_hash(config: dict, length: int = 10) -> str:
    config_str = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:length]


def cache_fci(
        molecule: str,
        geometry_fn: Callable[[float], str],
        distances: np.ndarray,
        basis: str = "sto-3g",
        active_space: Optional[Tuple[int, int]] = None,
    active_orbitals: Optional[Sequence[int]] = None,
        homo_lumo_window: int = 2,
        freeze_core: int = 0,
        data_dir: Optional[Path] = None,
        overwrite: bool = False,
        verbose: bool = True,
):
    # --- Setup paths ---
    if data_dir is None:
        data_dir = get_project_root() / "data"

    path = data_dir / molecule / basis
    path.mkdir(parents=True, exist_ok=True)

    # -- Experiment signature
    config = _build_cache_config(
        molecule=molecule,
        basis=basis,
        active_space=active_space,
        active_orbitals=tuple(active_orbitals) if active_orbitals is not None else None,
        homo_lumo_window=homo_lumo_window,
        freeze_core=freeze_core,
    )
    config_hash = _stable_config_hash(config)

    file_path = path / f"fci_{config_hash}.csv"
    meta_path = path / f"fci_{config_hash}.json"

    # --- Load existing cache ---
    done = {}

    if file_path.exists() and not overwrite:
        with open(file_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    d = float(row["distance"])
                    e = float(row["energy"])
                except (TypeError, ValueError, KeyError):
                    # Ignore malformed rows (e.g., duplicated headers inside CSV)
                    # to keep cache loading resilient.
                    continue
                done[d] = e

    # --- Save metadata (once) ---
    if not meta_path.exists() or overwrite:
        with open(meta_path, "w") as f:
            json.dump(config, f, indent=4)
    
    # --- Prepare file ---
    write_header = not file_path.exists() or overwrite

    file_mode = "w" if overwrite else "a"
    f = open(file_path, file_mode, newline="")
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
                    active_orbitals=active_orbitals,
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


def deduplicate_fci_cache(
        data_dir: Optional[Path] = None,
        dry_run: bool = True,
        verbose: bool = True,
) -> dict:
    """Remove arquivos de cache duplicados, mantendo o mais recente por assinatura de experimento.

    A assinatura é inferida do arquivo .json pareado (molecule, basis, active_space,
    homo_lumo_window, freeze_core). Arquivos sem metadata válida são preservados.
    """
    if data_dir is None:
        data_dir = get_project_root() / "data"

    kept_csv = []
    removed_csv = []
    kept_json = []
    removed_json = []
    skipped_csv = []

    grouped = {}
    for csv_path in data_dir.rglob("fci_*.csv"):
        json_path = csv_path.with_suffix(".json")
        if not json_path.exists():
            skipped_csv.append(str(csv_path))
            continue

        try:
            with open(json_path, "r") as f:
                meta = json.load(f)
        except Exception:
            skipped_csv.append(str(csv_path))
            continue

        try:
            active_space = meta.get("active_space")
            if isinstance(active_space, list):
                active_space = tuple(active_space)

            key_config = _build_cache_config(
                molecule=meta["molecule"],
                basis=meta["basis"],
                active_space=active_space,
                active_orbitals=tuple(meta.get("active_orbitals", []) or []) or None,
                homo_lumo_window=meta.get("homo_lumo_window", 2),
                freeze_core=meta.get("freeze_core", 0),
            )
            signature = _stable_config_hash(key_config)
        except Exception:
            skipped_csv.append(str(csv_path))
            continue

        grouped.setdefault(signature, []).append((csv_path, json_path))

    for _, items in grouped.items():
        # Keep newest CSV for each signature.
        newest_csv, newest_json = max(items, key=lambda pair: pair[0].stat().st_mtime)
        kept_csv.append(str(newest_csv))
        kept_json.append(str(newest_json))

        for csv_path, json_path in items:
            if csv_path == newest_csv:
                continue
            removed_csv.append(str(csv_path))
            removed_json.append(str(json_path))

            if not dry_run:
                if csv_path.exists():
                    csv_path.unlink()
                if json_path.exists():
                    json_path.unlink()

    summary = {
        "mode": "dry_run" if dry_run else "apply",
        "groups": len(grouped),
        "kept_csv": len(kept_csv),
        "removed_csv": len(removed_csv),
        "removed_json": len(removed_json),
        "skipped_csv": len(skipped_csv),
        "kept_files": kept_csv,
        "removed_files": removed_csv,
        "skipped_files": skipped_csv,
    }

    if verbose:
        print(json.dumps(summary, indent=2))

    return summary
