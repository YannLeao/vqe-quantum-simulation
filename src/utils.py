from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def get_project_root() -> Path:
    """
    Locate project root directory dynamically.
    Assumes notebooks/ exists inside project root.
    """
    return Path.cwd().resolve().parent


def plot_vqe_convergence(
        csv_path: str,
        title: str,
        output_name: str,
):
    """
    Generate convergence plot from VQE optimization history.
    """

    root = get_project_root()

    data_path = root / csv_path
    figures_dir = root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    history = pd.read_csv(data_path)

    plt.figure()
    plt.plot(history["iteration"], history["energy"])
    plt.xlabel("Iteration")
    plt.ylabel("Energy (Hartree)")
    plt.title(title)
    plt.grid(True)

    plt.savefig(
        figures_dir / output_name,
        dpi=300,
        bbox_inches="tight",
        )

    plt.close()