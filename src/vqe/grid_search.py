import hashlib
import json
import time
from itertools import product
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from qiskit.primitives import BaseEstimatorV2, StatevectorEstimator

from src.data.paths import get_vqe_cache_dir
from src.pipeline import run_experiment
from src.vqe.molecular_system import MolecularSystem

HARTREE_TO_KCAL_MOL = 627.509
CHEMICAL_ACCURACY_KCAL_MOL = 1.0


def run_vqe_grid_search(
        systems: list[MolecularSystem],
        parameter_grid: dict[str, Iterable[Any]],
        estimator: Optional[BaseEstimatorV2] = None,
        cache: bool = True,
        overwrite: bool = False,
        include_fci_reference: bool = True,
        chemical_accuracy_kcal_mol: float = CHEMICAL_ACCURACY_KCAL_MOL,
        run_label: str = "statevector",
        run_metadata: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    """Run a VQE grid search and cache one CSV/JSON pair per molecule/basis."""
    estimator = estimator or StatevectorEstimator()
    params_list = _expand_parameter_grid(parameter_grid)
    all_rows: list[dict[str, Any]] = []

    for system in systems:
        signature_payload = {
            "system": {
                "name": system.name,
                "basis": system.basis,
                "distances": system.distances,
                "active_space": system.active_space,
                "active_orbitals": system.active_orbitals,
                "homo_lumo_window": system.homo_lumo_window,
                "freeze_core": system.freeze_core,
            },
            "parameter_grid": parameter_grid,
            "estimator": estimator.__class__.__name__,
            "run_label": run_label,
            "run_metadata": run_metadata or {},
            "include_fci_reference": include_fci_reference,
            "chemical_accuracy_kcal_mol": chemical_accuracy_kcal_mol,
        }
        run_hash = _stable_hash(signature_payload)
        cache_dir = get_vqe_cache_dir(system.name, system.basis, create=True)
        csv_path = cache_dir / f"vqe_grid_{run_label}_{run_hash}.csv"
        json_path = cache_dir / f"vqe_grid_{run_label}_{run_hash}.json"

        if cache and csv_path.exists() and not overwrite:
            all_rows.extend(pd.read_csv(csv_path).to_dict("records"))
            continue

        references = (
            _compute_fci_references(system) if include_fci_reference else {}
        )

        rows = []
        for distance in system.distances:
            for params in params_list:
                config = _build_experiment_config(system, distance, params)
                try:
                    result = run_experiment(config, estimator=estimator)
                except Exception as exc:
                    result = {
                        "energy": np.nan,
                        "vqe_raw_energy": np.nan,
                        "eval_count": 0,
                        "optimal_params": None,
                        "history": [],
                        "success": False,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "timings": {},
                        "config": config.copy(),
                        "metadata": {},
                    }
                reference = references.get(float(distance), {})
                rows.append(
                    _result_to_row(
                        result,
                        reference_energy=reference.get("energy"),
                        reference_method=reference.get("method"),
                        chemical_accuracy_kcal_mol=chemical_accuracy_kcal_mol,
                        run_label=run_label,
                        run_metadata=run_metadata,
                    )
                )

        df = pd.DataFrame(rows)
        all_rows.extend(rows)

        if cache:
            df.to_csv(csv_path, index=False)
            metadata = {
                **signature_payload,
                "created_at": time.time(),
                "csv": str(csv_path),
            }
            json_path.write_text(json.dumps(metadata, indent=2, default=_json_default))

    return pd.DataFrame(all_rows)


def _expand_parameter_grid(grid: dict[str, Iterable[Any]]) -> list[dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in product(*(grid[key] for key in keys))]


def _build_experiment_config(system: MolecularSystem, distance: float | int, params: dict[str, Any]) -> dict[str, Any]:
    return {
        "molecule": system.name,
        "basis": system.basis,
        "distance": distance,
        "geometry": system.geometry_fn(distance),
        "active_space": system.active_space,
        "active_orbitals": system.active_orbitals,
        "homo_lumo_window": system.homo_lumo_window,
        "freeze_core": system.freeze_core,
        **params,
    }


def _result_to_row(
        result: dict[str, Any],
        reference_energy: Optional[float] = None,
        reference_method: Optional[str] = None,
        chemical_accuracy_kcal_mol: float = CHEMICAL_ACCURACY_KCAL_MOL,
        run_label: str = "statevector",
        run_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    config = result["config"]
    timings = result.get("timings", {})
    metadata = result.get("metadata", {})
    energy = result.get("energy")
    abs_error_hartree = None
    abs_error_kcal_mol = None
    within_chemical_accuracy = None

    if (
        energy is not None
        and reference_energy is not None
        and not pd.isna(energy)
        and not pd.isna(reference_energy)
    ):
        abs_error_hartree = abs(float(energy) - float(reference_energy))
        abs_error_kcal_mol = abs_error_hartree * HARTREE_TO_KCAL_MOL
        within_chemical_accuracy = abs_error_kcal_mol <= chemical_accuracy_kcal_mol

    return {
        "molecule": config["molecule"],
        "run_label": run_label,
        "noise_source": (run_metadata or {}).get("noise_source"),
        "backend_name": (run_metadata or {}).get("backend_name"),
        "shots": (run_metadata or {}).get("shots"),
        "basis": config["basis"],
        "distance": config["distance"],
        "mapper": config.get("mapper", "jw"),
        "ansatz": config["ansatz"],
        "reps": config.get("reps", 1),
        "optimizer": config["optimizer"],
        "max_iter": config.get("max_iter"),
        "seed": config.get("seed", 137),
        "active_space": json.dumps(config.get("active_space")),
        "active_orbitals": json.dumps(config.get("active_orbitals")),
        "freeze_core": config.get("freeze_core", False),
        "energy": energy,
        "reference_energy": reference_energy,
        "reference_method": reference_method,
        "fci_energy": reference_energy,
        "abs_error_hartree": abs_error_hartree,
        "abs_error_kcal_mol": abs_error_kcal_mol,
        "within_chemical_accuracy": within_chemical_accuracy,
        "vqe_raw_energy": result.get("vqe_raw_energy"),
        "eval_count": result.get("eval_count"),
        "success": result.get("success"),
        "error": result.get("error"),
        "error_type": result.get("error_type"),
        "num_qubits": metadata.get("num_qubits"),
        "num_terms": metadata.get("num_terms"),
        "constant_energy": metadata.get("constant_energy"),
        "setup_hamiltonian_s": timings.get("setup_hamiltonian"),
        "vqe_execution_s": timings.get("vqe_execution"),
        "total_experiment_s": timings.get("total_experiment"),
    }


def _compute_fci_references(system: MolecularSystem) -> dict[float, dict[str, Any]]:
    from src.data.cache import cache_fci

    distances = np.array(system.distances, dtype=float)
    energies = cache_fci(
        molecule=system.name,
        geometry_fn=system.geometry_fn,
        distances=distances,
        basis=system.basis,
        active_space=system.active_space,
        active_orbitals=system.active_orbitals,
        homo_lumo_window=system.homo_lumo_window,
        freeze_core=int(bool(system.freeze_core)),
        verbose=False,
    )
    method = "FCI" if system.active_space is None and system.homo_lumo_window == 0 else "CASCI"

    return {
        float(distance): {
            "energy": float(energy) if not pd.isna(energy) else np.nan,
            "method": method,
        }
        for distance, energy in zip(distances, energies)
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if callable(value):
        return getattr(value, "__name__", repr(value))
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _stable_hash(payload: dict[str, Any], length: int = 12) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]
