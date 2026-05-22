from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from qiskit.quantum_info import Statevector
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver

from src.vqe.ansatz import build_ansatz
from src.vqe.hamiltonian import (
    build_electronic_hamiltonian,
    build_electronic_problem,
    build_qubit_hamiltonian,
)
from src.vqe.molecular_system import MolecularSystem
from src.vqe.optimizer import get_optimizer
from src.vqe.vqe_runner import run_vqe


@dataclass(frozen=True)
class FourierCoefficients:
    a0_half: float
    ak: np.ndarray
    bk: np.ndarray


@dataclass(frozen=True)
class FourierLineResult:
    theta_grid: np.ndarray
    energy: np.ndarray
    coefficients: FourierCoefficients
    center: np.ndarray
    direction: np.ndarray


def wrap_pi(x: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi]."""
    return ((np.asarray(x, dtype=float) + np.pi) % (2 * np.pi)) - np.pi


def build_fourier_problem(
    system: MolecularSystem,
    distance: float,
    mapper: str = "parity",
    z2symmetry_reduction: bool = True,
):
    """Build a qubit Hamiltonian and constant energy for a molecular point."""
    atom = system.geometry_fn(float(distance))
    problem = build_electronic_problem(
        atom_string=atom,
        basis=system.basis,
        active_space=system.active_space,
        active_orbitals=system.active_orbitals,
        homo_lumo_window=system.homo_lumo_window,
        freeze_core=system.freeze_core,
    )
    fermionic_op, constant_energy = build_electronic_hamiltonian(
        atom_string=atom,
        basis=system.basis,
        active_space=system.active_space,
        active_orbitals=system.active_orbitals,
        homo_lumo_window=system.homo_lumo_window,
        freeze_core=system.freeze_core,
    )
    qubit_op = build_qubit_hamiltonian(
        fermionic_op,
        mapper=mapper,
        z2symmetry_reduction=z2symmetry_reduction,
        problem=problem,
        num_particles=problem.num_particles,
    ).simplify(atol=0.0)

    return problem, qubit_op, float(constant_energy)


def energy_line(
    ansatz,
    qubit_op,
    constant_energy: float,
    center: np.ndarray,
    direction: np.ndarray,
    theta_grid: Iterable[float],
) -> np.ndarray:
    """Evaluate E(theta) along p(theta)=center+theta*direction."""
    params = list(ansatz.parameters)
    center = np.asarray(center, dtype=float)
    direction = np.asarray(direction, dtype=float)
    values: list[float] = []

    for theta in theta_grid:
        point = wrap_pi(center + float(theta) * direction)
        bind = {params[i]: float(point[i]) for i in range(len(params))}
        state = Statevector.from_instruction(ansatz.assign_parameters(bind))
        values.append(float(np.real(state.expectation_value(qubit_op)) + constant_energy))

    return np.asarray(values, dtype=float)


def fit_fourier_series(energy_samples: np.ndarray) -> FourierCoefficients:
    """Fit real Fourier coefficients from equally spaced samples on [0, 2pi)."""
    y = np.asarray(energy_samples, dtype=float)
    n = len(y)
    fft_r = np.fft.rfft(y) / n

    return FourierCoefficients(
        a0_half=float(fft_r[0].real),
        ak=2.0 * np.real(fft_r[1:]),
        bk=-2.0 * np.imag(fft_r[1:]),
    )


def fourier_reconstruct(
    theta_grid: np.ndarray,
    coefficients: FourierCoefficients,
    n_harmonics: Optional[int] = None,
) -> np.ndarray:
    """Reconstruct a truncated Fourier series."""
    theta_grid = np.asarray(theta_grid, dtype=float)
    kmax = len(coefficients.ak) if n_harmonics is None else int(n_harmonics)
    kmax = min(kmax, len(coefficients.ak))
    reconstruction = np.full(theta_grid.shape, coefficients.a0_half, dtype=float)

    for k in range(1, kmax + 1):
        reconstruction += (
            coefficients.ak[k - 1] * np.cos(k * theta_grid)
            + coefficients.bk[k - 1] * np.sin(k * theta_grid)
        )

    return reconstruction


def spectral_metrics(coefficients: FourierCoefficients) -> dict[str, float]:
    """Return first-harmonic concentration and normalized spectral entropy."""
    amp2 = coefficients.ak.astype(float) ** 2 + coefficients.bk.astype(float) ** 2
    total = float(np.sum(amp2))
    if total <= 0.0:
        return {"r1": 1.0, "h_norm": 0.0}

    probabilities = amp2 / total
    entropy = float(-np.sum(probabilities * np.log(probabilities + 1e-16)))
    h_norm = float(entropy / np.log(len(probabilities))) if len(probabilities) > 1 else 0.0
    r1 = float(probabilities[0]) if len(probabilities) > 0 else 1.0

    return {"r1": r1, "h_norm": h_norm}


def run_vqe_reference_point(
    qubit_op,
    ansatz,
    constant_energy: float,
    optimizer_name: str = "cobyla",
    max_iter: int = 300,
    seed: int = 137,
) -> dict:
    """Run VQE once to obtain a center point for local Fourier analysis."""
    optimizer = get_optimizer(optimizer_name, max_iter=max_iter)
    return run_vqe(
        qubit_op=qubit_op,
        ansatz=ansatz,
        optimizer=optimizer,
        constant_energy=constant_energy,
        seed=seed,
    )


def make_coordinate_direction(num_parameters: int, parameter_index: int = 0) -> np.ndarray:
    direction = np.zeros(int(num_parameters), dtype=float)
    direction[int(parameter_index)] = 1.0
    return direction


def analyze_fourier_line(
    ansatz,
    qubit_op,
    constant_energy: float,
    center: np.ndarray,
    direction: np.ndarray,
    theta_samples: int = 64,
) -> FourierLineResult:
    """Sample and fit a one-dimensional Fourier cut of the VQE landscape."""
    theta_grid = np.linspace(0.0, 2.0 * np.pi, int(theta_samples), endpoint=False)
    samples = energy_line(ansatz, qubit_op, constant_energy, center, direction, theta_grid)
    coefficients = fit_fourier_series(samples)

    return FourierLineResult(
        theta_grid=theta_grid,
        energy=samples,
        coefficients=coefficients,
        center=np.asarray(center, dtype=float),
        direction=np.asarray(direction, dtype=float),
    )


def estimate_first_harmonic_guided_point(
    ansatz,
    qubit_op,
    constant_energy: float,
    center: np.ndarray,
    direction: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, int, dict[str, float]]:
    """Estimate a guided point using three probes of the first harmonic."""
    if direction is None:
        direction = make_coordinate_direction(ansatz.num_parameters, 0)

    probe_thetas = np.asarray([0.0, np.pi / 2.0, -np.pi / 2.0], dtype=float)
    probes = energy_line(ansatz, qubit_op, constant_energy, center, direction, probe_thetas)

    f0, f_plus, f_minus = map(float, probes)
    a0 = 0.5 * (f_plus + f_minus)
    b1 = 0.5 * (f_plus - f_minus)
    a1 = f0 - a0
    theta_star = float(np.arctan2(b1, a1) + np.pi)
    guided_point = wrap_pi(np.asarray(center, dtype=float) + theta_star * np.asarray(direction, dtype=float))

    return guided_point, len(probe_thetas), {
        "a1": float(a1),
        "b1": float(b1),
        "theta_star": theta_star,
        "first_harmonic_amplitude": float(np.hypot(a1, b1)),
    }


def scan_spectral_profile(
    systems: list[MolecularSystem],
    ansatz_name: str = "real_amplitudes",
    reps: int = 2,
    mapper: str = "parity",
    z2symmetry_reduction: bool = True,
    optimizer_name: str = "cobyla",
    max_iter: int = 300,
    theta_samples: int = 64,
    seed: int = 137,
    global_samples: int = 4,
    max_harmonics: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect local/global Fourier spectral metrics for several systems."""
    rng = np.random.default_rng(seed)
    metric_rows: list[dict] = []
    profile_rows: list[dict] = []

    for system in systems:
        for distance in system.distances:
            problem, qubit_op, constant_energy = build_fourier_problem(
                system,
                float(distance),
                mapper=mapper,
                z2symmetry_reduction=z2symmetry_reduction,
            )
            ansatz = build_ansatz(
                name=ansatz_name,
                num_qubits=qubit_op.num_qubits,
                reps=reps,
                num_particles=problem.num_particles,
                num_spatial_orbitals=problem.num_spatial_orbitals,
            )
            if ansatz.num_parameters == 0:
                continue

            vqe_result = run_vqe_reference_point(
                qubit_op=qubit_op,
                ansatz=ansatz,
                constant_energy=constant_energy,
                optimizer_name=optimizer_name,
                max_iter=max_iter,
                seed=seed,
            )
            if not vqe_result.get("success", False):
                continue

            center = np.asarray(vqe_result["optimal_params"], dtype=float)
            directions = [("local", make_coordinate_direction(ansatz.num_parameters, 0))]
            for _ in range(global_samples):
                direction = rng.normal(size=ansatz.num_parameters)
                direction = direction / (np.linalg.norm(direction) + 1e-12)
                directions.append(("global", direction))

            for regime, direction in directions:
                line = analyze_fourier_line(
                    ansatz=ansatz,
                    qubit_op=qubit_op,
                    constant_energy=constant_energy,
                    center=center if regime == "local" else rng.uniform(-np.pi, np.pi, ansatz.num_parameters),
                    direction=direction,
                    theta_samples=theta_samples,
                )
                metrics = spectral_metrics(line.coefficients)
                metric_rows.append({
                    "molecule": system.name,
                    "basis": system.basis,
                    "distance": float(distance),
                    "regime": regime,
                    **metrics,
                })

                amplitudes = np.sqrt(line.coefficients.ak ** 2 + line.coefficients.bk ** 2)
                amplitudes = amplitudes / (np.sum(amplitudes) + 1e-16)
                for k, amplitude in enumerate(amplitudes[:max_harmonics], start=1):
                    profile_rows.append({
                        "molecule": system.name,
                        "basis": system.basis,
                        "distance": float(distance),
                        "regime": regime,
                        "k": int(k),
                        "normalized_amplitude": float(amplitude),
                    })

    return pd.DataFrame(metric_rows), pd.DataFrame(profile_rows)


def scan_harmonic_error(
    systems: list[MolecularSystem],
    harmonic_grid: Iterable[int] = (1, 2, 3),
    ansatz_name: str = "real_amplitudes",
    reps: int = 2,
    mapper: str = "parity",
    z2symmetry_reduction: bool = True,
    optimizer_name: str = "cobyla",
    max_iter: int = 300,
    theta_samples: int = 64,
    seed: int = 137,
) -> pd.DataFrame:
    """Measure reconstruction and minimum-location error as K increases."""
    rows: list[dict] = []

    for system in systems:
        for distance in system.distances:
            problem, qubit_op, constant_energy = build_fourier_problem(
                system,
                float(distance),
                mapper=mapper,
                z2symmetry_reduction=z2symmetry_reduction,
            )
            ansatz = build_ansatz(
                name=ansatz_name,
                num_qubits=qubit_op.num_qubits,
                reps=reps,
                num_particles=problem.num_particles,
                num_spatial_orbitals=problem.num_spatial_orbitals,
            )
            if ansatz.num_parameters == 0:
                continue

            vqe_result = run_vqe_reference_point(
                qubit_op=qubit_op,
                ansatz=ansatz,
                constant_energy=constant_energy,
                optimizer_name=optimizer_name,
                max_iter=max_iter,
                seed=seed,
            )
            if not vqe_result.get("success", False):
                continue

            center = np.asarray(vqe_result["optimal_params"], dtype=float)
            direction = make_coordinate_direction(ansatz.num_parameters, 0)
            line = analyze_fourier_line(
                ansatz=ansatz,
                qubit_op=qubit_op,
                constant_energy=constant_energy,
                center=center,
                direction=direction,
                theta_samples=theta_samples,
            )
            true_min = float(np.min(line.energy))

            for k in harmonic_grid:
                reconstruction = fourier_reconstruct(line.theta_grid, line.coefficients, n_harmonics=int(k))
                idx = int(np.argmin(reconstruction))
                rows.append({
                    "molecule": system.name,
                    "basis": system.basis,
                    "distance": float(distance),
                    "K": int(k),
                    "rmse": float(np.sqrt(np.mean((line.energy - reconstruction) ** 2))),
                    "delta_min_energy": float(abs(float(line.energy[idx]) - true_min)),
                })

    return pd.DataFrame(rows)


def run_budget_comparison(
    systems: list[MolecularSystem],
    iteration_grid: Iterable[int] = (50, 100, 200, 400),
    ansatz_name: str = "real_amplitudes",
    reps: int = 2,
    mapper: str = "parity",
    z2symmetry_reduction: bool = True,
    optimizer_name: str = "cobyla",
    seed: int = 137,
    repeats: int = 3,
) -> pd.DataFrame:
    """Compare random initialization with first-harmonic Fourier guidance.

    Each repeat starts from the same random center for the default and guided
    modes. The guided mode spends three extra energy evaluations to estimate the
    first harmonic and move from that center.
    """
    rows: list[dict] = []
    rng = np.random.default_rng(seed)

    for max_iter in iteration_grid:
        for system in systems:
            for distance in system.distances:
                problem, qubit_op, constant_energy = build_fourier_problem(
                    system,
                    float(distance),
                    mapper=mapper,
                    z2symmetry_reduction=z2symmetry_reduction,
                )
                ansatz = build_ansatz(
                    name=ansatz_name,
                    num_qubits=qubit_op.num_qubits,
                    reps=reps,
                    num_particles=problem.num_particles,
                    num_spatial_orbitals=problem.num_spatial_orbitals,
                )
                if ansatz.num_parameters == 0:
                    continue

                exact = NumPyMinimumEigensolver().compute_minimum_eigenvalue(qubit_op)
                exact_total = float(np.real(exact.eigenvalue) + constant_energy)

                for repeat in range(int(repeats)):
                    center = rng.uniform(-np.pi, np.pi, size=ansatz.num_parameters)
                    guided_point, guide_cost, guide_info = estimate_first_harmonic_guided_point(
                        ansatz=ansatz,
                        qubit_op=qubit_op,
                        constant_energy=constant_energy,
                        center=center,
                    )

                    modes = [
                        ("random", center, 0, {}),
                        ("fourier_guided", guided_point, guide_cost, guide_info),
                    ]
                    for mode, initial_point, guidance_cost, info in modes:
                        optimizer = get_optimizer(optimizer_name, max_iter=int(max_iter))
                        result = run_vqe(
                            qubit_op=qubit_op,
                            ansatz=ansatz,
                            optimizer=optimizer,
                            initial_point=np.asarray(initial_point, dtype=float),
                            constant_energy=constant_energy,
                            seed=seed + repeat,
                        )
                        if not result.get("success", False):
                            continue

                        eval_count = int(result.get("eval_count", 0))
                        rows.append({
                            "molecule": system.name,
                            "basis": system.basis,
                            "distance": float(distance),
                            "repeat": int(repeat),
                            "max_iter": int(max_iter),
                            "mode": mode,
                            "eval_count": eval_count,
                            "guidance_cost": int(guidance_cost),
                            "total_cost": float(eval_count + guidance_cost),
                            "energy": float(result["energy"]),
                            "exact_energy": exact_total,
                            "abs_error": float(abs(float(result["energy"]) - exact_total)),
                            **info,
                        })

    return pd.DataFrame(rows)
