from qiskit_algorithms.optimizers import COBYLA


def get_optimizer(max_iter: int = 200) -> COBYLA:
    """
    Cria o otimizador clássico utilizado no VQE.

    O COBYLA é um método livre de gradiente adequado para
    otimização variacional em algoritmos quânticos híbridos.

    Parameters
    ----------
    max_iter : int, optional
        Número máximo de iterações do otimizador.

    Returns
    -------
    COBYLA
        Instância configurada do otimizador.
    """

    return COBYLA(maxiter=max_iter)
