from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from src.fourier.analysis import FourierLineResult, fourier_reconstruct


def plot_fourier_reconstruction(
    line: FourierLineResult,
    harmonic_orders: tuple[int, ...] = (1, 2, 3),
    ax=None,
):
    """Plot sampled E(theta) and truncated Fourier reconstructions."""
    ax = ax or plt.subplots(figsize=(8, 5))[1]
    ax.plot(line.theta_grid, line.energy, color="#111111", linewidth=2.4, label="E(theta)")

    markers = ["--", "-.", ":"]
    for i, order in enumerate(harmonic_orders):
        reconstruction = fourier_reconstruct(line.theta_grid, line.coefficients, n_harmonics=order)
        ax.plot(
            line.theta_grid,
            reconstruction,
            markers[i % len(markers)],
            linewidth=2,
            label=f"Fourier K={order}",
        )

    ax.set_title("Reconstrução de Fourier da paisagem VQE")
    ax.set_xlabel("theta")
    ax.set_ylabel("Energia (Hartree)")
    ax.grid(alpha=0.25)
    ax.legend()
    return ax


def plot_harmonic_profile(profile_df: pd.DataFrame, ax=None):
    """Plot mean normalized Fourier amplitude by harmonic index."""
    ax = ax or plt.subplots(figsize=(8, 5))[1]
    summary = (
        profile_df.groupby(["molecule", "regime", "k"], as_index=False)
        .agg(mean_amplitude=("normalized_amplitude", "mean"))
        .sort_values(["molecule", "regime", "k"])
    )

    for (molecule, regime), group in summary.groupby(["molecule", "regime"]):
        linestyle = "-" if regime == "local" else "--"
        marker = "o" if regime == "local" else "s"
        ax.plot(
            group["k"],
            group["mean_amplitude"],
            linestyle=linestyle,
            marker=marker,
            linewidth=2,
            label=f"{molecule} - {regime}",
        )

    ax.set_title("Perfil espectral médio")
    ax.set_xlabel("Harmônico k")
    ax.set_ylabel("Amplitude normalizada média")
    ax.grid(alpha=0.25)
    ax.legend()
    return ax


def plot_spectral_metrics(metric_df: pd.DataFrame, ax=None):
    """Plot R1 and normalized entropy against distance."""
    if ax is None:
        _, ax = plt.subplots(1, 2, figsize=(12, 4))

    for metric, axis, title, ylabel in [
        ("r1", ax[0], "Concentração no primeiro harmônico", "R1"),
        ("h_norm", ax[1], "Entropia espectral normalizada", "H_norm"),
    ]:
        summary = (
            metric_df.groupby(["molecule", "distance", "regime"], as_index=False)
            .agg(value=(metric, "mean"))
            .sort_values(["molecule", "regime", "distance"])
        )
        for (molecule, regime), group in summary.groupby(["molecule", "regime"]):
            linestyle = "-" if regime == "local" else "--"
            marker = "o" if regime == "local" else "s"
            axis.plot(
                group["distance"],
                group["value"],
                linestyle=linestyle,
                marker=marker,
                linewidth=2,
                label=f"{molecule} - {regime}",
            )

        axis.set_title(title)
        axis.set_xlabel("Distância (Å)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)

    ax[1].legend(fontsize=8)
    return ax


def plot_harmonic_error(error_df: pd.DataFrame, ax=None):
    """Plot reconstruction error as a function of Fourier order K."""
    if ax is None:
        _, ax = plt.subplots(1, 2, figsize=(12, 4))

    summary = (
        error_df.groupby(["molecule", "K"], as_index=False)
        .agg(
            mean_rmse=("rmse", "mean"),
            mean_delta_min=("delta_min_energy", "mean"),
        )
        .sort_values(["molecule", "K"])
    )

    for molecule, group in summary.groupby("molecule"):
        ax[0].plot(group["K"], group["mean_rmse"], marker="o", linewidth=2, label=molecule)
        ax[1].plot(group["K"], group["mean_delta_min"], marker="o", linewidth=2, label=molecule)

    ax[0].set_title("Erro de reconstrução vs K")
    ax[0].set_xlabel("Ordem harmônica K")
    ax[0].set_ylabel("RMSE médio")
    _set_log_scale_when_positive(ax[0], summary["mean_rmse"])
    ax[0].grid(alpha=0.25)

    ax[1].set_title("Erro no mínimo estimado vs K")
    ax[1].set_xlabel("Ordem harmônica K")
    ax[1].set_ylabel("Delta energia mínima médio")
    _set_log_scale_when_positive(ax[1], summary["mean_delta_min"])
    ax[1].grid(alpha=0.25)
    ax[1].legend()
    return ax


def plot_budget_comparison(comparison_df: pd.DataFrame, ax=None):
    """Plot Fourier-guided gain across optimizer budgets."""
    if ax is None:
        _, ax = plt.subplots(1, 2, figsize=(12, 4))

    summary = (
        comparison_df.groupby(["max_iter", "molecule", "mode"], as_index=False)
        .agg(
            mean_total_cost=("total_cost", "mean"),
            mean_abs_error=("abs_error", "mean"),
        )
    )
    wide = summary.pivot_table(
        index=["max_iter", "molecule"],
        columns="mode",
        values=["mean_total_cost", "mean_abs_error"],
        aggfunc="first",
    ).reset_index()
    wide.columns = [
        "max_iter",
        "molecule",
        "error_random",
        "error_guided",
        "cost_random",
        "cost_guided",
    ]
    wide["relative_error_reduction"] = 1.0 - (
        wide["error_guided"] / (wide["error_random"] + 1e-16)
    )
    wide["cost_delta"] = wide["cost_guided"] - wide["cost_random"]

    for molecule, group in wide.groupby("molecule"):
        group = group.sort_values("max_iter")
        ax[0].plot(group["max_iter"], group["relative_error_reduction"], marker="o", linewidth=2, label=molecule)
        ax[1].plot(group["max_iter"], group["cost_delta"], marker="o", linewidth=2, label=molecule)

    ax[0].axhline(0.0, color="#444444", linewidth=1, alpha=0.5)
    ax[0].set_title("Ganho de erro vs orçamento")
    ax[0].set_xlabel("Máximo de iterações")
    ax[0].set_ylabel("1 - erro_guiado / erro_padrão")
    ax[0].grid(alpha=0.25)

    ax[1].axhline(0.0, color="#444444", linewidth=1, alpha=0.5)
    ax[1].set_title("Custo guiado - custo padrão")
    ax[1].set_xlabel("Máximo de iterações")
    ax[1].set_ylabel("Delta de avaliações")
    ax[1].grid(alpha=0.25)
    ax[1].legend()
    return ax


def save_figure(fig, output_dir: str | Path, filename: str) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    target = path / filename
    fig.savefig(target, dpi=300, bbox_inches="tight")
    return target


def _set_log_scale_when_positive(ax, values: pd.Series):
    positive_values = pd.to_numeric(values, errors="coerce")
    positive_values = positive_values[positive_values > 0]
    if len(positive_values) > 0:
        ax.set_yscale("log")
    else:
        ax.set_yscale("linear")
