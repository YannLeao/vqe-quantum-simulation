from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B, SLSQP, SPSA, SciPyOptimizer, Optimizer, NELDER_MEAD


def get_optimizer(name: str, max_iter: int = 200) -> SciPyOptimizer | Optimizer:
    """Create a Qiskit optimizer by project-level name.

    Parameters
    ----------
    name:
        Optimizer identifier. Supported values are ``"cobyla"``,
        ``"l_bfgs_b"``, ``"spsa"``, ``"slsqp"``, and ``"nelder_mead"``.
    max_iter:
        Maximum number of optimizer iterations/evaluations passed to the
        corresponding Qiskit optimizer constructor.

    Returns
    -------
    SciPyOptimizer | Optimizer
        Configured Qiskit optimizer instance.

    Raises
    ------
    ValueError
        If ``name`` is not one of the supported optimizer identifiers.
    """

    name = name.lower()

    if name == "cobyla":
        return COBYLA(maxiter=max_iter)
    elif name == "l_bfgs_b":
        return L_BFGS_B(maxiter=max_iter)
    elif name == "spsa":
        return SPSA(maxiter=max_iter)
    elif name == "slsqp":
        return SLSQP(maxiter=max_iter)
    elif name == "nelder_mead":
        return NELDER_MEAD(maxiter=max_iter)
    else:
        raise ValueError(f"Unknown optimizer: {name}")
