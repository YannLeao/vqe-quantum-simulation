import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


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

def plot_fci(molecule:str, basis:str, show: bool = True):

    basis = basis.lower()
    path = f"../data/{molecule}/{basis}/fci.csv"

    df = pd.read_csv(path)

    plt.plot(df["distance"], df["fci_energy"], label=basis.upper())
    plt.xlabel("Distance (Å)")
    plt.ylabel("Energy (Hartree)")
    plt.title(f"{molecule} FCI Energy Curve")

    if show:
        plt.legend()
        plt.show()
