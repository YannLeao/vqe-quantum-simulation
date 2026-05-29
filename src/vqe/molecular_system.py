from dataclasses import dataclass
from typing import Callable, Optional

GeometryFn = Callable[[float], str]


@dataclass(frozen=True)
class MolecularSystem:
    name: str
    basis: str
    geometry_fn: GeometryFn
    distances: tuple[float, ...]
    active_space: Optional[tuple[int, int]] = None
    active_orbitals: Optional[tuple[int, ...]] = None
    homo_lumo_window: int = 0
    freeze_core: bool = False


def h2_geometry(distance: float | int) -> str:
    return f"H 0 0 0; H 0 0 {distance}"


def lih_geometry(distance: float | int) -> str:
    return f"Li 0 0 0; H 0 0 {distance}"


def li2o_linear_geometry(distance: float | int) -> str:
    return f"Li 0 0 {-distance}; O 0 0 0; Li 0 0 {distance}"


def beh2_linear_geometry(distance: float | int) -> str:
    return f"Be 0 0 0; H 0 0 {distance}; H 0 0 {-distance}"


def default_statevector_systems() -> list[MolecularSystem]:
    """Small default systems for a first noiseless VQE grid search."""
    return [
        MolecularSystem(
            name="H2",
            basis="sto-3g",
            geometry_fn=h2_geometry,
            distances=(0.5, 0.735, 1.0),
            freeze_core=False,
        ),
        MolecularSystem(
            name="LiH",
            basis="sto-3g",
            geometry_fn=lih_geometry,
            distances=(1.2, 1.5949, 2.2),
            active_space=(2, 3),
            active_orbitals=(0, 1, 2),
            freeze_core=True,
        ),
        MolecularSystem(
            name="Li2O_linear",
            basis="sto-3g",
            geometry_fn=li2o_linear_geometry,
            distances=(1.4, 1.8, 2.2),
            active_space=(4, 4),
            active_orbitals=(0, 1, 2, 3),
            freeze_core=True,
        ),
        MolecularSystem(
            name="BeH2",
            basis="sto-3g",
            geometry_fn=beh2_linear_geometry,
            distances=(0.8, 1.4, 2.0),
            active_space=(4, 4),
            active_orbitals=(0, 1, 2, 3),
            freeze_core=True,
        ),
    ]


def statevector_grid_systems(profile: str = "full") -> list[MolecularSystem]:
    """Return molecular systems for noiseless VQE grid-search experiments.

    Parameters
    ----------
    profile:
        `pilot` keeps the grid small for smoke tests. `full` uses more
        distances and basis sets while keeping active spaces modest enough for
        local StatevectorEstimator runs.
    """
    profile = profile.lower()

    if profile == "pilot":
        return default_statevector_systems()

    if profile != "full":
        raise ValueError("profile must be either 'pilot' or 'full'")

    h2_distances = tuple(float(x) for x in (0.35, 0.50, 0.65, 0.735, 0.90, 1.10, 1.40, 1.80, 2.20))
    lih_distances = tuple(float(x) for x in (1.00, 1.20, 1.40, 1.5949, 1.80, 2.10, 2.40, 2.80, 3.20))
    li2o_distances = tuple(float(x) for x in (1.20, 1.50, 1.80, 2.10, 2.40, 2.80, 3.20))
    beh2_distances = tuple(float(x) for x in (0.50, 0.80, 1.10, 1.40, 1.70, 2.00, 2.30, 2.60, 3.00))

    systems: list[MolecularSystem] = []

    for basis in ("sto-3g", "6-31g"):
        systems.append(
            MolecularSystem(
                name="H2",
                basis=basis,
                geometry_fn=h2_geometry,
                distances=h2_distances,
                freeze_core=False,
            )
        )

    for basis in ("sto-3g", "6-31g"):
        systems.append(
            MolecularSystem(
                name="LiH",
                basis=basis,
                geometry_fn=lih_geometry,
                distances=lih_distances,
                active_space=(2, 3),
                active_orbitals=(0, 1, 2),
                freeze_core=True,
            )
        )

    for basis in ("sto-3g", "6-31g"):
        systems.append(
            MolecularSystem(
                name="Li2O_linear",
                basis=basis,
                geometry_fn=li2o_linear_geometry,
                distances=li2o_distances,
                active_space=(4, 4),
                active_orbitals=(0, 1, 2, 3),
                freeze_core=True,
            )
        )

    for basis in ("sto-3g", "6-31g"):
        systems.append(
            MolecularSystem(
                name="BeH2",
                basis=basis,
                geometry_fn=beh2_linear_geometry,
                distances=beh2_distances,
                active_space=(4, 4),
                active_orbitals=(0, 1, 2, 3),
                freeze_core=True,
            )
        )

    return systems
