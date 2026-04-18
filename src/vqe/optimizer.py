from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B, SLSQP, SPSA, SciPyOptimizer, Optimizer, NELDER_MEAD


def get_optimizer(name: str, max_iter: int = 200) -> SciPyOptimizer | Optimizer:

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
