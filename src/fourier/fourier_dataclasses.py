from dataclasses import dataclass

@dataclass(frozen=True)
class FourierCoefficients:
    """Real Fourier coefficients fitted from an equally spaced energy line.

    Attributes
    ----------
    a0_half:
        Constant term of the real Fourier expansion. The name follows the
        common notation where the series is written as ``a0 / 2 + ...``.
    ak:
        Cosine coefficients ordered by harmonic index, where ``ak[0]`` is the
        coefficient of ``cos(theta)``.
    bk:
        Sine coefficients ordered by harmonic index, where ``bk[0]`` is the
        coefficient of ``sin(theta)``.
    """

    a0_half: float
    ak: np.ndarray
    bk: np.ndarray

@dataclass(frozen=True)
class FourierLineResult:
    """Sampled one-dimensional Fourier cut of a VQE energy landscape.

    Attributes
    ----------
    theta_grid:
        Angles, in radians, where the energy line was evaluated.
    energy:
        Total molecular energies evaluated at ``theta_grid``. The values
        include the Hamiltonian constant energy shift.
    coefficients:
        Real Fourier coefficients fitted from ``energy``.
    center:
        Parameter-space point used as the origin of the line.
    direction:
        Unit or coordinate direction used to define
        ``parameters(theta) = center + theta * direction``.
    """

    theta_grid: np.ndarray
    energy: np.ndarray
    coefficients: FourierCoefficients
    center: np.ndarray
    direction: np.ndarray