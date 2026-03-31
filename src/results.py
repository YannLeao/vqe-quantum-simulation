from pathlib import Path
from typing import List
import pandas as pd

from src.utils import get_project_root


def get_results_dir() -> Path:
    """
    Retorna o diretório destinado ao armazenamento dos resultados
    experimentais, criando-o automaticamente caso não exista.
    """
    root = get_project_root()
    results_dir = root / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def save_summary(
        filename: str,
        molecule: str,
        energy: float,
        fci: float,
        iterations: int,
) -> None:
    """
    Salva um resumo do experimento VQE em formato CSV.

    Parameters
    ----------
    filename : str
        Nome do arquivo de saída.
    molecule : str
        Identificação da molécula simulada.
    energy : float
        Energia estimada pelo VQE.
    fci : float
        Energia de referência obtida via Full Configuration Interaction.
    iterations : int
        Número total de iterações do processo variacional.
    """

    results_dir = get_results_dir()

    error = abs(energy - fci)

    df = pd.DataFrame(
        [
            {
                "molecule": molecule,
                "vqe_energy": energy,
                "fci_energy": fci,
                "absolute_error": error,
                "iterations": iterations,
            }
        ]
    )

    df.to_csv(results_dir / filename, index=False)


def save_history(
        filename: str,
        history: List[float],
) -> None:
    """
    Armazena o histórico de energias obtidas durante a
    otimização variacional do algoritmo VQE.

    Parameters
    ----------
    filename : str
        Nome do arquivo CSV de saída.
    history : List[float]
        Lista contendo os valores de energia registrados
        a cada iteração do algoritmo.
    """

    results_dir = get_results_dir()

    df = pd.DataFrame(
        {
            "iteration": range(1, len(history) + 1),
            "energy": history,
        }
    )

    df.to_csv(results_dir / filename, index=False)