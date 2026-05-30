from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import BaseEstimatorV2
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info import Statevector
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit_nature.second_q.problems import ElectronicStructureProblem

from src.fourier.fourier_dataclasses import FourierCoefficients, FourierLineResult
from src.vqe.ansatz import build_ansatz, decompose_for_estimator
from src.vqe.hamiltonian import (
    build_electronic_problem,
    build_qubit_hamiltonian,
)
from src.vqe.molecular_system import MolecularSystem
from src.vqe.optimizer import get_optimizer
from src.vqe.vqe_runner import run_vqe


def wrap_pi(x: np.ndarray) -> np.ndarray:
    """Wrap angles to the interval ``[-pi, pi]``.

    Parameters
    ----------
    x:
        Scalar-like or array-like angle values in radians.

    Returns
    -------
    np.ndarray
        Wrapped angles with the same broadcasted shape as ``x``.
    """
    return ((np.asarray(x, dtype=float) + np.pi) % (2 * np.pi)) - np.pi


def build_fourier_problem(
    system: MolecularSystem,
    distance: float,
    mapper: str = "parity",
    z2symmetry_reduction: bool = True,
) -> tuple[ElectronicStructureProblem, SparsePauliOp, float]:
    """Build the electronic problem used by the Fourier analysis.

    Parameters
    ----------
    system:
        Molecular system descriptor. It provides the geometry builder, basis
        set, active-space options, and distance grid.
    distance:
        Internuclear distance passed to ``system.geometry_fn``.
    mapper:
        Fermion-to-qubit mapper name accepted by
        :func:`src.vqe.hamiltonian.build_qubit_hamiltonian`. Current project
        values are ``"jw"``, ``"bk"``, and ``"parity"``.
    z2symmetry_reduction:
        Whether to request Qiskit Nature tapered mapping. This is usually used
        with the parity mapper to reduce qubit count when symmetries are
        available.

    Returns
    -------
    tuple[ElectronicStructureProblem, SparsePauliOp, float]
        The transformed electronic problem, the mapped qubit Hamiltonian, and
        the scalar energy shift that must be added to estimator eigenvalues.
    """
    atom = system.geometry_fn(float(distance))
    problem = build_electronic_problem(
        atom_string=atom,
        basis=system.basis,
        active_space=system.active_space,
        active_orbitals=system.active_orbitals,
        homo_lumo_window=system.homo_lumo_window,
        freeze_core=system.freeze_core,
    )
    fermionic_op = problem.hamiltonian.second_q_op()
    constant_energy = float(
        sum(float(np.real(v)) for v in problem.hamiltonian.constants.values())
    )
    qubit_op = build_qubit_hamiltonian(
        fermionic_op,
        mapper=mapper,
        z2symmetry_reduction=z2symmetry_reduction,
        problem=problem,
        num_particles=problem.num_particles,
    )

    return problem, qubit_op, float(constant_energy)


def energy_line(
    ansatz: QuantumCircuit,
    qubit_op: SparsePauliOp,
    constant_energy: float,
    center: np.ndarray,
    direction: np.ndarray,
    theta_grid: Iterable[float],
    estimator: Optional[BaseEstimatorV2] = None,
) -> np.ndarray:
    """Evaluate a one-dimensional VQE energy line.

    The sampled line is defined by
    ``parameters(theta) = wrap_pi(center + theta * direction)``. If
    ``estimator`` is omitted, the value is computed exactly with
    :class:`qiskit.quantum_info.Statevector`. If an estimator is provided, the
    ansatz is decomposed before submitting jobs to Aer/Qiskit primitives.

    Parameters
    ----------
    ansatz:
        Parameterized VQE circuit. Its number of parameters must match
        ``center`` and ``direction``.
    qubit_op:
        Qubit Hamiltonian whose expectation value defines the VQE objective.
    constant_energy:
        Scalar energy shift added to every expectation value.
    center:
        Parameter-space origin for the line.
    direction:
        Direction vector in parameter space. It may be a coordinate direction
        or a normalized random direction.
    theta_grid:
        Angles, in radians, used to sample the line.
    estimator:
        Optional Qiskit Estimator V2 primitive. Use ``None`` for exact
        statevector evaluation.

    Returns
    -------
    np.ndarray
        Total energies evaluated along the requested line.
    """
    params = list(ansatz.parameters)
    center = np.asarray(center, dtype=float)
    direction = np.asarray(direction, dtype=float)
    estimator_ansatz = decompose_for_estimator(ansatz) if estimator is not None else ansatz
    values: list[float] = []

    for theta in theta_grid:
        point = wrap_pi(center + float(theta) * direction)
        if estimator is None:
            bind = {params[i]: float(point[i]) for i in range(len(params))}
            state = Statevector.from_instruction(ansatz.assign_parameters(bind))
            energy = float(np.real(state.expectation_value(qubit_op)) + constant_energy)
        else:
            job = estimator.run([(estimator_ansatz, qubit_op, point)])
            ev = job.result()[0].data.evs
            if np.ndim(ev) > 0:
                ev = float(np.asarray(ev, dtype=float)[0])
            energy = float(ev + constant_energy)
        values.append(energy)

    return np.asarray(values, dtype=float)


def fit_fourier_series(energy_samples: np.ndarray) -> FourierCoefficients:
    """Fit real Fourier coefficients from equally spaced energy samples.

    Parameters
    ----------
    energy_samples:
        Values of a periodic function sampled uniformly on ``[0, 2*pi)``.

    Returns
    -------
    FourierCoefficients
        Real sine/cosine coefficients obtained from the real FFT.
    """
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
    """Reconstruct a Fourier approximation on a given theta grid.

    Parameters
    ----------
    theta_grid:
        Angles, in radians, where the approximation should be evaluated.
    coefficients:
        Fourier coefficients produced by :func:`fit_fourier_series`.
    n_harmonics:
        Number of harmonics to keep. Use ``None`` to include all available
        coefficients.

    Returns
    -------
    np.ndarray
        Reconstructed function values at ``theta_grid``.
    """
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
    """Compute compact spectral descriptors for a Fourier line.

    Parameters
    ----------
    coefficients:
        Fitted Fourier coefficients.

    Returns
    -------
    dict[str, float]
        Dictionary with ``r1`` and ``h_norm``. ``r1`` is the fraction of
        spectral power in the first harmonic. ``h_norm`` is the normalized
        entropy of the harmonic power distribution.
    """
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
    qubit_op: SparsePauliOp,
    ansatz: QuantumCircuit,
    constant_energy: float,
    optimizer_name: str = "cobyla",
    max_iter: int = 300,
    seed: int = 137,
    estimator: Optional[BaseEstimatorV2] = None,
) -> dict[str, Any]:
    """Run VQE once to obtain a center point for local Fourier analysis.

    Parameters
    ----------
    qubit_op:
        Qubit Hamiltonian optimized by VQE.
    ansatz:
        Parameterized ansatz circuit.
    constant_energy:
        Scalar energy shift added to the VQE eigenvalue.
    optimizer_name:
        Name accepted by :func:`src.vqe.optimizer.get_optimizer`.
    max_iter:
        Maximum optimizer iterations.
    seed:
        Seed used for the random initial point when one is not supplied.
    estimator:
        Optional Qiskit Estimator V2 primitive. Use ``None`` for the default
        statevector estimator.

    Returns
    -------
    dict[str, Any]
        Result dictionary produced by :func:`src.vqe.vqe_runner.run_vqe`.
        On success, it includes ``optimal_params`` and ``energy``.
    """
    optimizer = get_optimizer(optimizer_name, max_iter=max_iter)
    return run_vqe(
        qubit_op=qubit_op,
        ansatz=ansatz,
        optimizer=optimizer,
        estimator=estimator,
        constant_energy=constant_energy,
        seed=seed,
    )


def make_coordinate_direction(num_parameters: int, parameter_index: int = 0) -> np.ndarray:
    """Create a coordinate basis direction in ansatz parameter space.

    Parameters
    ----------
    num_parameters:
        Total number of variational parameters.
    parameter_index:
        Index that receives value ``1``. All other entries are ``0``.

    Returns
    -------
    np.ndarray
        Direction vector with shape ``(num_parameters,)``.
    """
    direction = np.zeros(int(num_parameters), dtype=float)
    direction[int(parameter_index)] = 1.0
    return direction


def analyze_fourier_line(
    ansatz: QuantumCircuit,
    qubit_op: SparsePauliOp,
    constant_energy: float,
    center: np.ndarray,
    direction: np.ndarray,
    theta_samples: int = 64,
    estimator: Optional[BaseEstimatorV2] = None,
) -> FourierLineResult:
    """Sample and fit a one-dimensional Fourier cut of the VQE landscape.

    Parameters
    ----------
    ansatz:
        Parameterized ansatz circuit.
    qubit_op:
        Qubit Hamiltonian whose expectation value is sampled.
    constant_energy:
        Scalar energy shift added to all expectation values.
    center:
        Parameter vector around which the line is sampled.
    direction:
        Direction in parameter space.
    theta_samples:
        Number of equally spaced samples on ``[0, 2*pi)``.
    estimator:
        Optional Qiskit Estimator V2 primitive.

    Returns
    -------
    FourierLineResult
        Sampled energies, fitted coefficients, and the line definition.
    """
    theta_grid = np.linspace(0.0, 2.0 * np.pi, int(theta_samples), endpoint=False)
    samples = energy_line(ansatz, qubit_op, constant_energy, center, direction, theta_grid, estimator=estimator)
    coefficients = fit_fourier_series(samples)

    return FourierLineResult(
        theta_grid=theta_grid,
        energy=samples,
        coefficients=coefficients,
        center=np.asarray(center, dtype=float),
        direction=np.asarray(direction, dtype=float),
    )


def estimate_first_harmonic_guided_point(
    ansatz: QuantumCircuit,
    qubit_op: SparsePauliOp,
    constant_energy: float,
    center: np.ndarray,
    direction: Optional[np.ndarray] = None,
    estimator: Optional[BaseEstimatorV2] = None,
) -> tuple[np.ndarray, int, dict[str, float]]:
    """Estimate an initial point from a first-harmonic Fourier approximation.

    The method probes the energy at ``theta = 0``, ``pi/2``, and ``-pi/2``.
    Those three values determine the first sine/cosine harmonic and therefore
    the minimum of the approximation
    ``a0 + a1*cos(theta) + b1*sin(theta)``.

    Parameters
    ----------
    ansatz:
        Parameterized ansatz circuit.
    qubit_op:
        Qubit Hamiltonian whose expectation value is probed.
    constant_energy:
        Scalar energy shift added to all expectation values.
    center:
        Starting parameter vector.
    direction:
        Direction along which the first harmonic is estimated. When ``None``,
        the first coordinate direction is used.
    estimator:
        Optional Qiskit Estimator V2 primitive.

    Returns
    -------
    tuple[np.ndarray, int, dict[str, float]]
        Guided parameter vector, number of extra energy evaluations spent on
        the guidance step, and diagnostic Fourier values.
    """
    if direction is None:
        direction = make_coordinate_direction(ansatz.num_parameters, 0)

    probe_thetas = np.asarray([0.0, np.pi / 2.0, -np.pi / 2.0], dtype=float)
    probes = energy_line(ansatz, qubit_op, constant_energy, center, direction, probe_thetas, estimator=estimator)

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
    estimator: Optional[BaseEstimatorV2] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect local and global Fourier spectral summaries.

    For each molecular point, the function first obtains a local VQE optimum.
    It then compares one local coordinate line against several random global
    lines. This measures whether the energy landscape is Fourier-simple only
    near a good VQE solution or also across random regions of parameter space.

    Parameters
    ----------
    systems:
        Molecular systems to scan.
    ansatz_name:
        Ansatz name accepted by :func:`src.vqe.ansatz.build_ansatz`.
    reps:
        Repetition depth passed to hardware-efficient ansatz constructors.
    mapper:
        Fermion-to-qubit mapper name.
    z2symmetry_reduction:
        Whether to use Qiskit Nature symmetry tapering when building the qubit
        Hamiltonian.
    optimizer_name:
        Optimizer name used for the reference VQE center.
    max_iter:
        Maximum optimizer iterations for the reference VQE run.
    theta_samples:
        Number of samples per Fourier line.
    seed:
        Random seed used for global centers and directions.
    global_samples:
        Number of random global directions sampled per molecular point.
    max_harmonics:
        Maximum harmonic index stored in the profile output.
    estimator:
        Optional Qiskit Estimator V2 primitive.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``spectral_df`` with one row per line and metrics ``r1``/``h_norm``;
        ``profile_df`` with normalized amplitude by harmonic index.
    """
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
                estimator=estimator,
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
                    estimator=estimator,
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
    estimator: Optional[BaseEstimatorV2] = None,
) -> pd.DataFrame:
    """Measure Fourier reconstruction error as harmonic order increases.

    Parameters
    ----------
    systems:
        Molecular systems to scan.
    harmonic_grid:
        Harmonic truncation orders ``K`` to evaluate.
    ansatz_name:
        Ansatz name accepted by :func:`src.vqe.ansatz.build_ansatz`.
    reps:
        Repetition depth passed to hardware-efficient ansatz constructors.
    mapper:
        Fermion-to-qubit mapper name.
    z2symmetry_reduction:
        Whether to use Qiskit Nature symmetry tapering.
    optimizer_name:
        Optimizer used to obtain the local VQE center.
    max_iter:
        Maximum optimizer iterations for the reference VQE run.
    theta_samples:
        Number of samples per Fourier line.
    seed:
        Random seed used by the reference VQE run.
    estimator:
        Optional Qiskit Estimator V2 primitive.

    Returns
    -------
    pd.DataFrame
        Rows with ``rmse`` for full-curve reconstruction error and
        ``delta_min_energy`` for the error induced by the minimum predicted
        from a truncated Fourier reconstruction.
    """
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
                estimator=estimator,
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
                estimator=estimator,
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
    estimator: Optional[BaseEstimatorV2] = None,
) -> pd.DataFrame:
    """Compare random initialization with first-harmonic Fourier guidance.

    Each repeat starts from the same random center for the default and guided
    modes. The guided mode spends three extra energy evaluations to estimate the
    first harmonic and move from that center.

    Parameters
    ----------
    systems:
        Molecular systems and distance grids to evaluate.
    iteration_grid:
        Optimizer budgets used for both random and Fourier-guided starts.
    ansatz_name:
        Ansatz name accepted by :func:`src.vqe.ansatz.build_ansatz`.
    reps:
        Repetition depth passed to hardware-efficient ansatz constructors.
    mapper:
        Fermion-to-qubit mapper name.
    z2symmetry_reduction:
        Whether to use Qiskit Nature symmetry tapering.
    optimizer_name:
        Optimizer used in each VQE run.
    seed:
        Base seed for repeatable random centers.
    repeats:
        Number of random centers evaluated per molecular point and iteration
        budget.
    estimator:
        Optional Qiskit Estimator V2 primitive.

    Returns
    -------
    pd.DataFrame
        One row per successful VQE run. The ``mode`` column is either
        ``"random"`` or ``"fourier_guided"``; ``guidance_cost`` stores the
        extra evaluations spent by Fourier guidance.
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
                        estimator=estimator,
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
                            estimator=estimator,
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
