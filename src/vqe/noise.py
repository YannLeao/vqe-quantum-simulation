from dataclasses import dataclass
from os import getenv
from typing import Any, Optional


IBM_HERON_R2_BACKENDS = ("ibm_kingston", "ibm_marrakesh", "ibm_fez")


@dataclass(frozen=True)
class NoisyEstimatorConfig:
    backend_name: str
    shots: int = 4096
    seed_simulator: int = 137
    gate_error: bool = True
    readout_error: bool = True
    thermal_relaxation: bool = True

    def metadata(self) -> dict[str, Any]:
        return {
            "noise_source": "ibm_backend_noise_model",
            "backend_name": self.backend_name,
            "shots": self.shots,
            "seed_simulator": self.seed_simulator,
            "gate_error": self.gate_error,
            "readout_error": self.readout_error,
            "thermal_relaxation": self.thermal_relaxation,
        }


def get_ibm_service(channel: str = "ibm_quantum_platform"):
    """Return a Qiskit Runtime service, loading IBM_API_KEY from .env if present."""
    from dotenv import load_dotenv
    from qiskit_ibm_runtime import QiskitRuntimeService

    load_dotenv()
    token = getenv("IBM_API_KEY") or getenv("QISKIT_IBM_TOKEN")

    if token:
        return QiskitRuntimeService(channel=channel, token=token)

    return QiskitRuntimeService(channel=channel)


def get_ibm_backend(backend_name: str, channel: str = "ibm_quantum_platform"):
    """Return an IBM Quantum backend by name using saved Qiskit Runtime credentials."""
    service = get_ibm_service(channel=channel)
    return service.backend(backend_name)


def list_ibm_backends(channel: str = "ibm_quantum_platform") -> list[str]:
    """List backend names visible to the saved Qiskit Runtime account."""
    service = get_ibm_service(channel=channel)
    return sorted(backend.name for backend in service.backends())


def build_backend_noise_model(
    backend,
    gate_error: bool = True,
    readout_error: bool = True,
    thermal_relaxation: bool = True,
):
    """Build an Aer noise model from backend calibration/properties."""
    from qiskit_aer.noise import NoiseModel

    return NoiseModel.from_backend(
        backend,
        gate_error=gate_error,
        readout_error=readout_error,
        thermal_relaxation=thermal_relaxation,
    )


def build_noisy_aer_estimator(
    backend,
    shots: int = 4096,
    seed_simulator: int = 137,
    gate_error: bool = True,
    readout_error: bool = True,
    thermal_relaxation: bool = True,
):
    """Build an Aer EstimatorV2 configured with a backend-derived noise model."""
    from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2

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

    return AerEstimatorV2(
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
):
    """Fetch an IBM backend and build a local noisy Aer estimator from it."""
    backend = get_ibm_backend(backend_name)
    return build_noisy_aer_estimator(
        backend,
        shots=shots,
        seed_simulator=seed_simulator,
        gate_error=gate_error,
        readout_error=readout_error,
        thermal_relaxation=thermal_relaxation,
    )


def _backend_coupling_map(backend) -> Optional[list[list[int]]]:
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
