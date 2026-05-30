from dataclasses import dataclass
from math import sqrt
from os import getenv
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from qiskit_aer.noise import NoiseModel
    from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
    from qiskit_ibm_runtime import QiskitRuntimeService


IBM_HERON_R2_BACKENDS = ("ibm_kingston", "ibm_marrakesh", "ibm_fez")


@dataclass(frozen=True)
class NoisyEstimatorConfig:
    """Configuration for an Aer estimator built from IBM backend properties.

    Attributes
    ----------
    backend_name:
        IBM Quantum backend name used as the calibration source.
    shots:
        Number of shots requested from the Aer estimator.
    seed_simulator:
        Seed forwarded to the Aer simulator backend.
    gate_error:
        Whether to include backend gate errors in the noise model.
    readout_error:
        Whether to include backend readout errors in the noise model.
    thermal_relaxation:
        Whether to include backend thermal relaxation effects.
    """

    backend_name: str
    shots: int = 4096
    seed_simulator: int = 137
    gate_error: bool = True
    readout_error: bool = True
    thermal_relaxation: bool = True

    def metadata(self) -> dict[str, Any]:
        """Return cache metadata that identifies this backend-noise setup."""
        return {
            "noise_source": "ibm_backend_noise_model",
            "backend_name": self.backend_name,
            "shots": self.shots,
            "seed_simulator": self.seed_simulator,
            "gate_error": self.gate_error,
            "readout_error": self.readout_error,
            "thermal_relaxation": self.thermal_relaxation,
        }


@dataclass(frozen=True)
class SyntheticNoiseConfig:
    """Configuration for a lightweight synthetic Aer noise model.

    Attributes
    ----------
    shots:
        Number of shots requested from the Aer estimator. The value also
        determines ``default_precision`` as ``1 / sqrt(shots)``.
    seed_simulator:
        Seed forwarded to the Aer simulator backend.
    single_qubit_depolarizing:
        Depolarizing probability applied after supported one-qubit gates.
    two_qubit_depolarizing:
        Depolarizing probability applied after supported two-qubit gates.
    readout_error:
        Symmetric bit-flip probability applied during measurement readout.
    """

    shots: int = 2048
    seed_simulator: int = 137
    single_qubit_depolarizing: float = 0.001
    two_qubit_depolarizing: float = 0.01
    readout_error: float = 0.02

    @property
    def default_precision(self) -> float:
        """Estimator precision implied by the configured shot count."""
        return 1.0 / sqrt(float(self.shots))

    def metadata(self) -> dict[str, Any]:
        """Return cache metadata that uniquely identifies this noise setup."""
        return {
            "noise_source": "synthetic_depolarizing_readout",
            "noise_pipeline": "aer_estimator_decomposed_ansatz_v1",
            "shots": self.shots,
            "default_precision": self.default_precision,
            "seed_simulator": self.seed_simulator,
            "single_qubit_depolarizing": self.single_qubit_depolarizing,
            "two_qubit_depolarizing": self.two_qubit_depolarizing,
            "readout_error": self.readout_error,
        }


def build_synthetic_noise_model(
    single_qubit_depolarizing: float = 0.001,
    two_qubit_depolarizing: float = 0.01,
    readout_error: float = 0.02,
) -> "NoiseModel":
    """Build a synthetic depolarizing/readout noise model.

    Parameters
    ----------
    single_qubit_depolarizing:
        Depolarizing probability applied to ``id``, ``x``, ``sx``, ``h``,
        ``rx``, ``ry``, and ``rz`` gates. Use ``0`` to disable this channel.
    two_qubit_depolarizing:
        Depolarizing probability applied to ``cx`` gates. Use ``0`` to disable
        this channel.
    readout_error:
        Symmetric probability of reading ``0`` as ``1`` and ``1`` as ``0``.
        Use ``0`` to disable readout noise.

    Returns
    -------
    qiskit_aer.noise.NoiseModel
        Aer noise model suitable for ``AerEstimatorV2`` backend options.
    """
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    noise_model = NoiseModel()

    if single_qubit_depolarizing > 0:
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(single_qubit_depolarizing, 1),
            ["id", "x", "sx", "h", "rx", "ry", "rz"],
        )

    if two_qubit_depolarizing > 0:
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(two_qubit_depolarizing, 2),
            ["cx"],
        )

    if readout_error > 0:
        noise_model.add_all_qubit_readout_error(
            ReadoutError(
                [
                    [1.0 - readout_error, readout_error],
                    [readout_error, 1.0 - readout_error],
                ]
            )
        )

    return noise_model


def build_synthetic_noisy_aer_estimator(
    shots: int = 2048,
    seed_simulator: int = 137,
    single_qubit_depolarizing: float = 0.001,
    two_qubit_depolarizing: float = 0.01,
    readout_error: float = 0.02,
) -> "AerEstimatorV2":
    """Build an Aer Estimator V2 with synthetic depolarizing/readout noise.

    Parameters
    ----------
    shots:
        Number of shots requested by the estimator. The estimator precision is
        set to ``1 / sqrt(shots)``.
    seed_simulator:
        Random seed forwarded to the Aer simulator backend.
    single_qubit_depolarizing:
        One-qubit depolarizing probability.
    two_qubit_depolarizing:
        Two-qubit depolarizing probability.
    readout_error:
        Symmetric readout bit-flip probability.

    Returns
    -------
    qiskit_aer.primitives.EstimatorV2
        Configured Aer estimator ready to be passed to VQE or Fourier
        sampling utilities.
    """
    from qiskit_aer.primitives import EstimatorV2 as AerEstimator

    noise_model = build_synthetic_noise_model(
        single_qubit_depolarizing=single_qubit_depolarizing,
        two_qubit_depolarizing=two_qubit_depolarizing,
        readout_error=readout_error,
    )

    return AerEstimator(
        options={
            "default_precision": 1.0 / sqrt(float(shots)),
            "backend_options": {
                "noise_model": noise_model,
                "basis_gates": noise_model.basis_gates,
                "seed_simulator": seed_simulator,
            },
            "run_options": {"shots": shots},
        }
    )


def get_ibm_service(channel: str = "ibm_quantum_platform") -> "QiskitRuntimeService":
    """Create a Qiskit Runtime service from environment credentials.

    Parameters
    ----------
    channel:
        Qiskit Runtime channel. The IBM Quantum Platform channel is used by
        default.

    Returns
    -------
    qiskit_ibm_runtime.QiskitRuntimeService
        Runtime service configured from ``IBM_API_KEY``/``QISKIT_IBM_TOKEN``
        or from a saved local account.
    """
    from dotenv import load_dotenv
    from qiskit_ibm_runtime import QiskitRuntimeService

    load_dotenv()
    token = getenv("IBM_API_KEY") or getenv("QISKIT_IBM_TOKEN")

    if token:
        return QiskitRuntimeService(channel=channel, token=token)

    return QiskitRuntimeService(channel=channel)


def get_ibm_backend(backend_name: str, channel: str = "ibm_quantum_platform") -> Any:
    """Fetch an IBM Quantum backend by name.

    Parameters
    ----------
    backend_name:
        Backend name, for example ``"ibm_kingston"``.
    channel:
        Qiskit Runtime channel passed to :func:`get_ibm_service`.

    Returns
    -------
    qiskit_ibm_runtime.IBMBackend
        Backend object visible to the configured account.
    """
    service = get_ibm_service(channel=channel)
    return service.backend(backend_name)


def list_ibm_backends(channel: str = "ibm_quantum_platform") -> list[str]:
    """List IBM Quantum backend names visible to the configured account."""
    service = get_ibm_service(channel=channel)
    return sorted(backend.name for backend in service.backends())


def build_backend_noise_model(
    backend: Any,
    gate_error: bool = True,
    readout_error: bool = True,
    thermal_relaxation: bool = True,
) -> "NoiseModel":
    """Build an Aer noise model from backend calibration data.

    Parameters
    ----------
    backend:
        IBM backend object with calibration/properties understood by
        ``NoiseModel.from_backend``.
    gate_error:
        Whether to include gate depolarizing/error channels.
    readout_error:
        Whether to include readout assignment errors.
    thermal_relaxation:
        Whether to include thermal relaxation effects when available.

    Returns
    -------
    qiskit_aer.noise.NoiseModel
        Noise model derived from the backend.
    """
    from qiskit_aer.noise import NoiseModel

    return NoiseModel.from_backend(
        backend,
        gate_error=gate_error,
        readout_error=readout_error,
        thermal_relaxation=thermal_relaxation,
    )


def build_noisy_aer_estimator(
    backend: Any,
    shots: int = 4096,
    seed_simulator: int = 137,
    gate_error: bool = True,
    readout_error: bool = True,
    thermal_relaxation: bool = True,
) -> "AerEstimatorV2":
    """Build an Aer Estimator V2 from an IBM backend-derived noise model.

    Parameters
    ----------
    backend:
        IBM backend object used as the calibration source for
        ``NoiseModel.from_backend``.
    shots:
        Number of shots requested from the Aer estimator.
    seed_simulator:
        Random seed forwarded to the Aer simulator backend.
    gate_error:
        Whether to include backend gate errors.
    readout_error:
        Whether to include backend readout errors.
    thermal_relaxation:
        Whether to include backend thermal relaxation channels.

    Returns
    -------
    qiskit_aer.primitives.EstimatorV2
        Configured estimator using the backend-derived noise model.
    """
    from qiskit_aer.primitives import EstimatorV2 as AerEstimator

    noise_model = build_backend_noise_model(
        backend,
        gate_error=gate_error,
        readout_error=readout_error,
        thermal_relaxation=thermal_relaxation,
    )

    backend_options = {
        "noise_model": noise_model,
        "basis_gates": noise_model.basis_gates,
        "seed_simulator": seed_simulator,
    }

    coupling_map = _backend_coupling_map(backend)
    if coupling_map is not None:
        backend_options["coupling_map"] = coupling_map

    return AerEstimator(
        options={
            "backend_options": backend_options,
            "run_options": {"shots": shots, "seed_simulator": seed_simulator},
        }
    )


def build_noisy_aer_estimator_from_backend_name(
    backend_name: str,
    shots: int = 4096,
    seed_simulator: int = 137,
    gate_error: bool = True,
    readout_error: bool = True,
    thermal_relaxation: bool = True,
) -> "AerEstimatorV2":
    """Fetch an IBM backend and build a local noisy Aer estimator from it.

    Parameters
    ----------
    backend_name:
        IBM backend name, for example ``"ibm_kingston"``.
    shots:
        Number of shots requested from the Aer estimator.
    seed_simulator:
        Random seed forwarded to the Aer simulator backend.
    gate_error:
        Whether to include backend gate errors.
    readout_error:
        Whether to include backend readout errors.
    thermal_relaxation:
        Whether to include backend thermal relaxation channels.

    Returns
    -------
    qiskit_aer.primitives.EstimatorV2
        Configured estimator using a noise model built from the requested IBM
        backend.
    """
    backend = get_ibm_backend(backend_name)
    return build_noisy_aer_estimator(
        backend,
        shots=shots,
        seed_simulator=seed_simulator,
        gate_error=gate_error,
        readout_error=readout_error,
        thermal_relaxation=thermal_relaxation,
    )


def _backend_coupling_map(backend: Any) -> Optional[list[list[int]]]:
    """Extract a backend coupling map using either Runtime or legacy APIs."""
    if hasattr(backend, "coupling_map"):
        coupling_map = backend.coupling_map
        if coupling_map is not None:
            return coupling_map

    if hasattr(backend, "configuration"):
        try:
            return backend.configuration().coupling_map
        except Exception:
            return None

    return None
