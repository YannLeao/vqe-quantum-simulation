from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib import pyplot as plt

from src.data.paths import get_data_dir

CHEMICAL_ACCURACY_KCAL_MOL = 1.0

MOLECULE_LABELS = {
    "H2": "H2",
    "LiH": "LiH",
    "Li2O_linear": "Li2O linear",
    "BeH2": "BeH2",
}

MARKERS = ["o", "s", "^", "D", "X", "P", "v", "*"]


def load_latest_vqe_results(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load the newest VQE grid CSV for each molecule/basis/run label."""
    root = get_data_dir(data_dir)
    latest_files: dict[tuple[str, str, str], Path] = {}

    for path in root.rglob("vqe_cache/vqe_grid_*.csv"):
        rel = path.relative_to(root)
        if len(rel.parts) < 4:
            continue

        try:
            sample = pd.read_csv(path, nrows=1)
        except Exception:
            continue

        molecule, basis = rel.parts[0], rel.parts[1]
        run_label = (
            str(sample["run_label"].iloc[0])
            if "run_label" in sample.columns and not sample["run_label"].isna().all()
            else "statevector"
        )
        key = (molecule, basis, run_label)
        previous = latest_files.get(key)
        if previous is None or path.stat().st_mtime > previous.stat().st_mtime:
            latest_files[key] = path

    frames = []
    for path in latest_files.values():
        df = pd.read_csv(path)
        if "run_label" not in df.columns:
            df["run_label"] = "statevector"
        df["source_path"] = str(path)
        df["source_mtime"] = path.stat().st_mtime
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def load_latest_fci_curves(
    data_dir: Optional[Path] = None,
    strategy: str = "latest",
) -> pd.DataFrame:
    """Load one FCI/CASCI CSV for each molecule/basis pair.

    Parameters
    ----------
    strategy:
        `latest` selects the newest cache file. `densest` selects the file with
        the largest number of rows, which is useful for dissociation curves.
        :param strategy:
        :param data_dir:
    """
    root = get_data_dir(data_dir)
    latest_files: dict[tuple[str, str], Path] = {}
    row_counts: dict[tuple[str, str], int] = {}
    strategy = strategy.lower()

    if strategy not in {"latest", "densest"}:
        raise ValueError("strategy must be either 'latest' or 'densest'")

    for path in root.rglob("fci_*.csv"):
        rel = path.relative_to(root)
        if len(rel.parts) < 3:
            continue

        molecule, basis = rel.parts[0], rel.parts[1]
        key = (molecule, basis)
        previous = latest_files.get(key)

        if strategy == "latest":
            should_replace = previous is None or path.stat().st_mtime > previous.stat().st_mtime
        else:
            row_count = _csv_row_count(path)
            should_replace = previous is None or row_count > row_counts.get(key, -1)
            row_counts[key] = max(row_count, row_counts.get(key, -1))

        if should_replace:
            latest_files[key] = path

    frames = []
    for path in latest_files.values():
        df = pd.read_csv(path)
        rel = path.relative_to(root)
        df["molecule"] = rel.parts[0]
        df["basis"] = rel.parts[1]
        df["source_path"] = str(path)
        df["source_mtime"] = path.stat().st_mtime
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def _csv_row_count(path: Path) -> int:
    try:
        return max(0, sum(1 for _ in path.open("r", encoding="utf-8")) - 1)
    except UnicodeDecodeError:
        return max(0, sum(1 for _ in path.open("r")) - 1)


def best_vqe_by_point(
    vqe_df: pd.DataFrame,
    group_cols: Iterable[str] = ("molecule", "basis", "distance"),
) -> pd.DataFrame:
    """Return the best successful VQE row per point by absolute error."""
    if vqe_df.empty:
        return vqe_df.copy()

    df = vqe_df[vqe_df["success"].astype(bool)].copy()
    df = df.dropna(subset=["abs_error_kcal_mol"])

    return (
        df.sort_values("abs_error_kcal_mol")
        .groupby(list(group_cols), as_index=False)
        .head(1)
        .sort_values(list(group_cols))
        .reset_index(drop=True)
    )


def chemical_accuracy_config_table(
    vqe_df: pd.DataFrame,
    group_cols: Iterable[str] = ("molecule", "basis", "ansatz", "reps", "optimizer"),
    statevector_only: bool = True,
    only_accurate: bool = False,
    rank_scope: Iterable[str] = ("molecule",),
) -> pd.DataFrame:
    """Summarize statevector VQE configurations in a plain DataFrame."""
    if vqe_df.empty:
        return pd.DataFrame()

    df = vqe_df.copy()
    if statevector_only and "run_label" in df.columns:
        df = df[df["run_label"].fillna("statevector").astype(str).str.startswith("statevector")]

    df = df[_as_bool_series(df["success"])].copy()
    df = df.dropna(subset=["abs_error_kcal_mol"])
    if df.empty:
        return pd.DataFrame()

    df["within_chemical_accuracy"] = _as_bool_series(df["within_chemical_accuracy"])
    group_cols = list(group_cols)

    summary = (
        df.groupby(group_cols, as_index=False)
        .agg(
            points=("distance", "count"),
            accurate_points=("within_chemical_accuracy", "sum"),
            best_error_kcal_mol=("abs_error_kcal_mol", "min"),
            mean_error_kcal_mol=("abs_error_kcal_mol", "mean"),
            median_error_kcal_mol=("abs_error_kcal_mol", "median"),
            mean_time_s=("total_experiment_s", "mean"),
        )
    )

    best_rows = df.loc[df.groupby(group_cols)["abs_error_kcal_mol"].idxmin(), group_cols + ["distance"]]
    best_rows = best_rows.rename(columns={"distance": "best_distance"})
    summary = summary.merge(best_rows, on=group_cols, how="left")
    summary["accuracy_rate"] = summary["accurate_points"] / summary["points"]
    summary["reached_chemical_accuracy"] = summary["accurate_points"] > 0

    if only_accurate:
        summary = summary[summary["accurate_points"] > 0].copy()

    if summary.empty:
        return summary

    summary = summary.sort_values(
        ["accuracy_rate", "mean_error_kcal_mol", "best_error_kcal_mol", "mean_time_s"],
        ascending=[False, True, True, True],
        na_position="last",
    ).reset_index(drop=True)

    rank_scope = list(rank_scope)
    if rank_scope:
        summary["rank"] = summary.groupby(rank_scope, sort=False).cumcount().add(1)
    else:
        summary["rank"] = np.arange(1, len(summary) + 1)

    summary["is_best"] = summary["rank"] == 1
    summary["config"] = (
        summary["ansatz"].map(_pretty_label)
        + " | "
        + summary["optimizer"].str.upper()
        + " | reps="
        + summary["reps"].astype(str)
    )
    if "basis" in summary.columns:
        summary["config"] = summary["basis"] + " | " + summary["config"]

    return summary


def ensure_figures_dir(path: str | Path = "outputs/figures/slides") -> Path:
    """Create and return a figure output directory."""
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_figure(fig: Figure, output_dir: Path, filename: str) -> Path:
    """Save a Matplotlib figure as a high-resolution slide asset."""
    path = output_dir / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return path


def plot_dissociation_curve(
    fci_df: pd.DataFrame,
    vqe_df: pd.DataFrame,
    molecule: str,
    basis: str,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot FCI/CASCI and best VQE energies over distance."""
    ax = ax or plt.subplots(figsize=(8, 5))[1]
    fci = _filter(fci_df, molecule=molecule, basis=basis).sort_values("distance")
    best = best_vqe_by_point(_filter(vqe_df, molecule=molecule, basis=basis))

    if not fci.empty:
        method = fci["method"].iloc[0] if "method" in fci else "FCI/CASCI"
        ax.plot(
            fci["distance"],
            fci["energy"],
            color="#F0E442",
            linestyle="--",
            linewidth=2.5,
            label=f"{method}",
        )

    if not best.empty:
        ax.plot(
            best["distance"],
            best["energy"],
            color="#0072B2",
            marker="s",
            linewidth=2,
            label="VQE - Grid Search",
        )

    _style_axes(
        ax,
        title=f"Curva de dissociação - {MOLECULE_LABELS.get(molecule, molecule)} ({basis})",
        xlabel="Distância (Å)",
        ylabel="Energia (Hartree)",
    )
    ax.legend()
    return ax


def plot_ansatz_comparison(
    vqe_df: pd.DataFrame,
    molecule: str,
    basis: str,
    optimizer: Optional[str] = None,
    ax: Optional[Axes] = None,
) -> Axes:
    """Compare ansatz choices by best error at each distance."""
    ax = ax or plt.subplots(figsize=(8, 5))[1]
    df = _filter(vqe_df, molecule=molecule, basis=basis)
    if optimizer is not None:
        df = df[df["optimizer"] == optimizer]

    for i, (ansatz, group) in enumerate(sorted(df.groupby("ansatz"))):
        best = best_vqe_by_point(group, group_cols=("molecule", "basis", "distance", "ansatz"))
        ax.plot(
            best["distance"],
            best["abs_error_kcal_mol"],
            marker=MARKERS[i % len(MARKERS)],
            linewidth=2,
            label=_pretty_label(ansatz),
        )

    _add_chemical_accuracy_line(ax)
    suffix = f" - {optimizer.upper()}" if optimizer else ""
    _style_axes(
        ax,
        title=f"Comparação de ansatz - {MOLECULE_LABELS.get(molecule, molecule)} ({basis}){suffix}",
        xlabel="Distância (Å)",
        ylabel="Erro absoluto (kcal/mol)",
    )
    ax.legend()
    return ax


def plot_optimizer_comparison(
    vqe_df: pd.DataFrame,
    molecule: str,
    basis: str,
    ansatz: str,
    ax: Optional[Axes] = None,
) -> Axes:
    """Compare optimizers for a fixed molecule, basis, and ansatz."""
    ax = ax or plt.subplots(figsize=(8, 5))[1]
    df = _filter(vqe_df, molecule=molecule, basis=basis)
    df = df[df["ansatz"] == ansatz]

    for i, (optimizer, group) in enumerate(sorted(df.groupby("optimizer"))):
        best = best_vqe_by_point(group, group_cols=("molecule", "basis", "distance", "optimizer"))
        ax.plot(
            best["distance"],
            best["abs_error_kcal_mol"],
            marker=MARKERS[i % len(MARKERS)],
            linewidth=2,
            label=optimizer.upper(),
        )

    _add_chemical_accuracy_line(ax)
    _style_axes(
        ax,
        title=f"Comparação de otimizadores - {MOLECULE_LABELS.get(molecule, molecule)} ({basis}, {_pretty_label(ansatz)})",
        xlabel="Distância (Å)",
        ylabel="Erro absoluto (kcal/mol)",
    )
    ax.legend()
    return ax


def plot_chemical_accuracy_rate(vqe_df: pd.DataFrame, ax: Optional[Axes] = None) -> Axes:
    """Plot chemical accuracy rate by molecule and basis."""
    ax = ax or plt.subplots(figsize=(8, 5))[1]
    best = best_vqe_by_point(vqe_df)
    summary = (
        best.groupby(["molecule", "basis"], as_index=False)
        .agg(
            points=("distance", "count"),
            accurate=("within_chemical_accuracy", "sum"),
            mean_error=("abs_error_kcal_mol", "mean"),
        )
    )
    summary["rate"] = summary["accurate"] / summary["points"]
    summary["label"] = summary["molecule"].map(MOLECULE_LABELS).fillna(summary["molecule"]) + "\n" + summary["basis"]

    colors = np.where(summary["rate"] >= 0.8, "#009E73", "#D55E00")
    bars = ax.bar(summary["label"], summary["rate"], color=colors)
    ax.bar_label(bars, labels=[f"{value:.0%}" for value in summary["rate"]], padding=3)
    ax.set_ylim(0, 1.12)
    _style_axes(
        ax,
        title="Taxa de pontos dentro da precisão química",
        xlabel="Sistema",
        ylabel="Fração dos pontos",
    )
    return ax


def plot_runtime_by_configuration(vqe_df: pd.DataFrame, ax: Optional[Axes] = None) -> Axes:
    """Plot mean runtime by ansatz and optimizer."""
    ax = ax or plt.subplots(figsize=(8, 5))[1]
    df = vqe_df[vqe_df["success"].astype(bool)].copy()
    summary = (
        df.groupby(["ansatz", "optimizer"], as_index=False)
        .agg(mean_time=("total_experiment_s", "mean"))
        .sort_values("mean_time")
    )
    summary["label"] = summary["ansatz"].map(_pretty_label) + "\n" + summary["optimizer"].str.upper()

    bars = ax.barh(summary["label"], summary["mean_time"], color="#56B4E9")
    ax.bar_label(bars, labels=[f"{value:.1f}s" for value in summary["mean_time"]], padding=3, fontsize=9)
    _style_axes(
        ax,
        title="Tempo médio por configuração",
        xlabel="Tempo médio (s)",  # Agora o X mostra os segundos (horizontal)
        ylabel="Configuração",     # Agora o Y mostra os nomes (vertical)
    )
    return ax


def _filter(df: pd.DataFrame, molecule: str, basis: str) -> pd.DataFrame:
    return df[(df["molecule"] == molecule) & (df["basis"] == basis)].copy()


def _as_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)

    return (
        series.fillna(False)
        .astype(str)
        .str.lower()
        .isin({"true", "1", "yes"})
    )


def _add_chemical_accuracy_line(ax):
    ax.axhline(
        CHEMICAL_ACCURACY_KCAL_MOL,
        color="#D55E00",
        linestyle=":",
        linewidth=2,
        label="Precisão química (1 kcal/mol)",
    )


def _style_axes(ax, title: str, xlabel: str, ylabel: str):
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def _pretty_label(value: str) -> str:
    labels = {
        "real_amplitudes": "RealAmplitudes",
        "efficient_su2": "EfficientSU2",
        "uccsd": "UCCSD",
        "puccsd": "PUCCSD",
        "cobyla": "COBYLA",
        "slsqp": "SLSQP",
    }
    return labels.get(value, value)
