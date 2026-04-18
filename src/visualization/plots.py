import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.utils.paths import get_project_root


def plot_error(
        vqe_energies,
        fci_energies,
        kcal=False,
        chemistry_precision_line=False,
        distances=np.linspace(0.3, 2.5, 100),
        molecule_name="H2",

):
    vqe_energies = np.array(vqe_energies)
    fci_energies = np.array(fci_energies)

    if kcal:
        hartree_to_kcal = 627.509

        vqe_energies = np.array(vqe_energies) * hartree_to_kcal
        fci_energies = np.array(fci_energies) * hartree_to_kcal

    label = "(kcal/mol)" if kcal else "(Hartree)"

    error_abs = abs(vqe_energies - fci_energies)
    plt.figure(figsize=(10, 6))
    plt.plot(distances, error_abs, "o--", label="Erro absoluto ", linewidth=2, markersize=8)

    plt.xlabel("Distância interatômica (Å)", fontsize=12)
    plt.ylabel(f"Erro absoluto {label}", fontsize=12)
    plt.title(f"Curva de erro relativo do {molecule_name}", fontsize=14)

    if kcal and chemistry_precision_line:
        line = np.zeros(len(distances)) + 1
        plt.plot(distances, line, "--", label="Precisão química")

    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)

    plt.tight_layout()
    plt.show()

def plot_fci(
        molecule:str,
        basis:str,
        data_dir: Optional[Path] = None,
        normalize=False,
        show: bool = True
):

    if data_dir is None:
        data_dir = get_project_root() / "data"

    path = data_dir / molecule / basis

    files = list(path.glob("fci_*.csv"))

    if not files:
        raise FileNotFoundError(f"No FCI data found for {molecule} ({basis})")

    plt.figure()

    for file in files:
        df = pd.read_csv(file)
        df = df.sort_values("distance")
        method = df["method"].iloc[0]

        if normalize:
            r_eq = df.loc[df["energy"].idxmin(), "distance"]
            df["distance"] /= r_eq
            df["energy"] -= df["energy"].min()

        label = f"{basis.upper()} | {method}"

        plt.plot(df["distance"], df["energy"], label=label)

    plt.xlabel("Distance (Å)")
    plt.ylabel("Energy (Hartree)")
    plt.title(f"{molecule} FCI Energy Curve")
    plt.legend()

    if show:
        plt.show()