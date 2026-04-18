from typing import Tuple, Callable, Optional

import numpy as np

from src.solvers.fci import compute_fci_energy


def generate_distance_grid(
        geometry_fn: Callable[[float], str],
        basis: str = "sto-3g",
        active_space: Optional[Tuple[int, int]] = None,
        homo_lumo_window: int = 2,
        freeze_core: int = 0,
        coarse_range: Tuple[float, float] = (0.5, 5.0),
        coarse_points: int = 25,
        fine_range_factor: Tuple[float, float] = (0.5, 3.0),
        fine_points: int = 100,
        verbose: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Generates an adaptive interatomic distance grid based on equilibrium distance.

    Returns:
        distances (np.ndarray): refined grid
        r_eq (float): estimated equilibrium distance
    """

    # --- Step 1: coarse scan ---
    coarse_distances = np.linspace(*coarse_range, coarse_points)
    energies = []

    if verbose:
        print("Coarse scan...")

    for d in coarse_distances:
        try:
            e = compute_fci_energy(
                atom_string=geometry_fn(d),
                basis=basis,
                active_space=active_space,
                homo_lumo_window=homo_lumo_window,
                freeze_core=freeze_core,
            )

            if isinstance(e, np.ndarray):
                e = e.item()

            e = float(e)
        except Exception:
            e = np.nan
            print(f"[ERRO coarse] d={d:.3f} → {e}")

        energies.append(e)

    energies = np.array(energies, dtype=float)

    # --- Step 2: find equilibrium ---
    valid_mask = (~np.isnan(energies)) & (energies < 0)

    if not np.any(valid_mask):
        raise RuntimeError("All coarse scan points failed.")

    valid_distances = coarse_distances[valid_mask]
    valid_energies = energies[valid_mask]

    r_eq = valid_distances[np.argmin(valid_energies)]

    if verbose:
        print(f"Estimated equilibrium distance: {r_eq:.4f} Å")

    # --- Step 3: refined grid ---
    d_min = fine_range_factor[0] * r_eq
    d_max = fine_range_factor[1] * r_eq

    distances = np.linspace(d_min, d_max, fine_points)

    if verbose:
        print(f"Fine grid: [{d_min:.3f}, {d_max:.3f}] with {fine_points} points")

    return distances
