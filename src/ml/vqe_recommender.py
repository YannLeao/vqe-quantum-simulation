"""Grouped evaluation of machine-learning recommenders for VQE configs.

The prediction unit is one VQE configuration at one molecular geometry.  The
outer validation split keeps all candidate configurations from a geometry in
the same fold, preventing the model from seeing the same molecular point in
both training and evaluation data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


HARTREE_TO_KCAL_MOL = 627.509
CHEMICAL_ACCURACY_KCAL_MOL = 1.0


@dataclass(frozen=True)
class ExperimentConfig:
    """Control the grouped recommender experiment.

    Parameters
    ----------
    data_root:
        Repository data directory containing the canonical VQE cache layout.
    output_dir:
        Directory receiving CSV and JSON results. The function creates it.
    random_seed:
        Seed shared by estimators, random baselines, and bootstrap resampling.
    outer_splits:
        Number of geometry-grouped folds used for final evaluation.
    inner_splits:
        Number of geometry-grouped folds used to select hyperparameters using
        only each outer training partition.
    random_repeats:
        Number of random top-k selections evaluated per test geometry.
    bootstrap_repeats:
        Number of group-level bootstrap resamples used for confidence bounds.
    """

    data_root: Path
    output_dir: Path
    random_seed: int = 137
    outer_splits: int = 5
    inner_splits: int = 3
    random_repeats: int = 1000
    bootstrap_repeats: int = 1000


def load_vqe_grid_dataset(data_root: Path) -> pd.DataFrame:
    """Load and validate the completed STO-3G dense VQE grid-search caches.

    Returns
    -------
    pandas.DataFrame
        One row per VQE configuration and geometry, enriched with stable
        geometry/configuration identifiers and the logarithmic regression
        target.

    Raises
    ------
    FileNotFoundError
        If no canonical dense grid-search cache is found.
    ValueError
        If unsuccessful/non-finite rows exist or geometries do not contain the
        expected 18 candidate configurations.
    """

    files = sorted(data_root.glob("*/*/vqe_cache/vqe_dense_curve_config_search_sto3g_*.csv"))
    if not files:
        raise FileNotFoundError(f"No dense VQE grid caches found below {data_root}")

    data = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    required = {
        "molecule",
        "basis",
        "distance",
        "ansatz",
        "optimizer",
        "reps",
        "num_qubits",
        "num_terms",
        "abs_error_kcal_mol",
        "within_chemical_accuracy",
        "success",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"VQE dataset is missing columns: {missing}")

    success = data["success"].astype(bool)
    finite = np.isfinite(data["abs_error_kcal_mol"]) & np.isfinite(data["distance"])
    if not (success & finite).all():
        bad = data.loc[~(success & finite), ["molecule", "distance", "ansatz", "optimizer"]]
        raise ValueError(f"Dataset contains failed or non-finite rows:\n{bad.to_string(index=False)}")

    data = data.copy()
    data["reps"] = data["reps"].astype(int)
    data["geometry_id"] = data["molecule"].astype(str) + "@" + data["distance"].map(lambda x: f"{x:.12g}")
    data["config_id"] = (
        data["ansatz"].astype(str)
        + "|"
        + data["optimizer"].astype(str)
        + "|r="
        + data["reps"].astype(str)
    )
    data["target_log_error"] = np.log1p(data["abs_error_kcal_mol"].clip(lower=0.0))

    group_sizes = data.groupby("geometry_id").size()
    if not group_sizes.eq(18).all():
        raise ValueError(f"Every geometry must contain 18 candidates; observed {group_sizes.to_dict()}")
    if data["geometry_id"].nunique() != 27 or len(data) != 486:
        raise ValueError(
            "Expected the frozen ENIAC dataset with 486 rows and 27 geometries; "
            f"found {len(data)} rows and {data['geometry_id'].nunique()} geometries"
        )
    return data.sort_values(["molecule", "distance", "config_id"]).reset_index(drop=True)


def _preprocessor(include_molecule: bool) -> tuple[ColumnTransformer, list[str]]:
    categorical = ["ansatz", "optimizer"]
    if include_molecule:
        categorical.insert(0, "molecule")
    numeric = ["distance", "reps", "num_qubits", "num_terms"]
    features = categorical + numeric
    transformer = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", StandardScaler(), numeric),
        ]
    )
    return transformer, features


def _model_specs(seed: int, include_molecule: bool) -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    """Return deterministic estimators and compact hyperparameter grids."""

    specs: dict[str, tuple[Any, dict[str, list[Any]]]] = {
        "dummy": (DummyRegressor(strategy="mean"), {}),
        "ridge": (Ridge(), {"model__alpha": [0.1, 1.0, 10.0]}),
        "random_forest": (
            RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=1),
            {
                "model__max_depth": [None, 8],
                "model__min_samples_leaf": [1, 2],
            },
        ),
        "mlp": (
            MLPRegressor(
                activation="relu",
                early_stopping=True,
                max_iter=3000,
                n_iter_no_change=50,
                random_state=seed,
            ),
            {
                "model__hidden_layer_sizes": [(32, 16), (64, 32)],
                "model__alpha": [1e-3, 1e-2],
            },
        ),
    }

    result: dict[str, tuple[Pipeline, dict[str, list[Any]]]] = {}
    for name, (estimator, grid) in specs.items():
        preprocessor, _ = _preprocessor(include_molecule)
        result[name] = (Pipeline([("preprocessor", preprocessor), ("model", estimator)]), grid)
    return result


def _regression_metrics(predictions: pd.DataFrame, protocol: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, fold), frame in predictions.groupby(["model", "fold"]):
        true_error = frame["abs_error_kcal_mol"].to_numpy()
        pred_error = frame["predicted_error_kcal_mol"].to_numpy()
        true_log = frame["target_log_error"].to_numpy()
        pred_log = frame["predicted_log_error"].to_numpy()
        rows.append(
            {
                "protocol": protocol,
                "model": model,
                "fold": fold,
                "mae_kcal_mol": mean_absolute_error(true_error, pred_error),
                "rmse_kcal_mol": np.sqrt(mean_squared_error(true_error, pred_error)),
                "mae_log": mean_absolute_error(true_log, pred_log),
                "r2_log": r2_score(true_log, pred_log),
            }
        )
    return pd.DataFrame(rows)


def _fit_grouped_protocol(
    data: pd.DataFrame,
    config: ExperimentConfig,
    *,
    protocol: str,
    split_groups: pd.Series,
    include_molecule: bool,
    outer_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Fit all models with nested group-aware validation."""

    _, features = _preprocessor(include_molecule)
    outer_cv = GroupKFold(n_splits=outer_splits)
    predictions: list[pd.DataFrame] = []
    baseline_predictions: list[pd.DataFrame] = []
    selected_params: list[dict[str, Any]] = []

    for fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(data[features], data["target_log_error"], groups=split_groups), start=1
    ):
        train = data.iloc[train_idx].copy()
        test = data.iloc[test_idx].copy()
        inner_groups = train["geometry_id"]
        inner_cv = GroupKFold(n_splits=min(config.inner_splits, inner_groups.nunique()))

        for model_name, (pipeline, param_grid) in _model_specs(config.random_seed, include_molecule).items():
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring="neg_mean_absolute_error",
                cv=inner_cv,
                n_jobs=-1,
                refit=True,
            )
            search.fit(
                train[features],
                train["target_log_error"],
                groups=inner_groups,
            )
            pred_log = np.maximum(search.predict(test[features]), 0.0)
            frame = test.copy()
            frame["protocol"] = protocol
            frame["fold"] = fold
            frame["model"] = model_name
            frame["predicted_log_error"] = pred_log
            frame["predicted_error_kcal_mol"] = np.expm1(pred_log)
            predictions.append(frame)
            selected_params.append(
                {
                    "protocol": protocol,
                    "fold": fold,
                    "model": model_name,
                    "best_score_neg_mae_log": float(search.best_score_),
                    "best_params": search.best_params_,
                }
            )

        config_means = train.groupby("config_id")["target_log_error"].mean()
        global_frame = test.copy()
        global_frame["protocol"] = protocol
        global_frame["fold"] = fold
        global_frame["model"] = "global_best"
        global_frame["predicted_log_error"] = global_frame["config_id"].map(config_means)
        global_frame["predicted_error_kcal_mol"] = np.expm1(global_frame["predicted_log_error"])
        baseline_predictions.append(global_frame)

        if include_molecule:
            molecule_means = train.groupby(["molecule", "config_id"])["target_log_error"].mean()
            molecule_keys = pd.MultiIndex.from_frame(test[["molecule", "config_id"]])
            molecule_scores = molecule_means.reindex(molecule_keys).to_numpy()
            fallback_scores = test["config_id"].map(config_means).to_numpy()

            molecule_frame = test.copy()
            molecule_frame["protocol"] = protocol
            molecule_frame["fold"] = fold
            molecule_frame["model"] = "molecule_best"
            molecule_frame["predicted_log_error"] = np.where(
                np.isfinite(molecule_scores), molecule_scores, fallback_scores
            )
            molecule_frame["predicted_error_kcal_mol"] = np.expm1(
                molecule_frame["predicted_log_error"]
            )
            baseline_predictions.append(molecule_frame)

    prediction_df = pd.concat(predictions, ignore_index=True)
    baseline_df = pd.concat(baseline_predictions, ignore_index=True)
    return prediction_df, baseline_df, selected_params


def _top_k_rows(scored: pd.DataFrame, top_ks: tuple[int, ...] = (1, 3, 5)) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (protocol, model, fold, geometry_id), group in scored.groupby(
        ["protocol", "model", "fold", "geometry_id"]
    ):
        ordered = group.sort_values(["predicted_log_error", "config_id"])
        oracle_error = float(group["abs_error_kcal_mol"].min())
        for k in top_ks:
            selected = ordered.head(k)
            selected_error = float(selected["abs_error_kcal_mol"].min())
            rows.append(
                {
                    "protocol": protocol,
                    "model": model,
                    "fold": fold,
                    "geometry_id": geometry_id,
                    "molecule": group["molecule"].iloc[0],
                    "top_k": k,
                    "hit_chemical_accuracy": bool(selected["within_chemical_accuracy"].astype(bool).any()),
                    "best_error_in_top_k": selected_error,
                    "oracle_error": oracle_error,
                    "regret_kcal_mol": selected_error - oracle_error,
                    "evaluations_avoided_pct": 100.0 * (1.0 - k / len(group)),
                }
            )
    return pd.DataFrame(rows)


def _random_baseline(
    data: pd.DataFrame,
    config: ExperimentConfig,
    protocol: str = "geometry_grouped",
    top_ks: tuple[int, ...] = (1, 3, 5),
) -> pd.DataFrame:
    rng = np.random.default_rng(config.random_seed)
    rows: list[dict[str, Any]] = []
    for geometry_id, group in data.groupby("geometry_id"):
        oracle_error = float(group["abs_error_kcal_mol"].min())
        indices = np.arange(len(group))
        for k in top_ks:
            hits: list[float] = []
            selected_errors: list[float] = []
            for _ in range(config.random_repeats):
                selected = group.iloc[rng.choice(indices, size=k, replace=False)]
                hits.append(float(selected["within_chemical_accuracy"].astype(bool).any()))
                selected_errors.append(float(selected["abs_error_kcal_mol"].min()))
            rows.append(
                {
                    "protocol": protocol,
                    "model": "random",
                    "fold": 0,
                    "geometry_id": geometry_id,
                    "molecule": group["molecule"].iloc[0],
                    "top_k": k,
                    "hit_chemical_accuracy": float(np.mean(hits)),
                    "best_error_in_top_k": float(np.mean(selected_errors)),
                    "oracle_error": oracle_error,
                    "regret_kcal_mol": float(np.mean(selected_errors)) - oracle_error,
                    "evaluations_avoided_pct": 100.0 * (1.0 - k / len(group)),
                }
            )
    return pd.DataFrame(rows)


def _oracle_baseline(
    data: pd.DataFrame,
    protocol: str,
    top_ks: tuple[int, ...] = (1, 3, 5),
) -> pd.DataFrame:
    """Build the unattainable full-grid reference used to measure regret."""

    rows: list[dict[str, Any]] = []
    for geometry_id, group in data.groupby("geometry_id"):
        oracle_error = float(group["abs_error_kcal_mol"].min())
        for k in top_ks:
            rows.append(
                {
                    "protocol": protocol,
                    "model": "oracle",
                    "fold": 0,
                    "geometry_id": geometry_id,
                    "molecule": group["molecule"].iloc[0],
                    "top_k": k,
                    "hit_chemical_accuracy": oracle_error <= CHEMICAL_ACCURACY_KCAL_MOL,
                    "best_error_in_top_k": oracle_error,
                    "oracle_error": oracle_error,
                    "regret_kcal_mol": 0.0,
                    "evaluations_avoided_pct": 100.0 * (1.0 - k / len(group)),
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_summary(rows: pd.DataFrame, repeats: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    summaries: list[dict[str, Any]] = []
    for (protocol, model, top_k), frame in rows.groupby(["protocol", "model", "top_k"]):
        records = frame.reset_index(drop=True)
        samples: list[tuple[float, float, float]] = []
        for _ in range(repeats):
            sampled = records.iloc[rng.integers(0, len(records), size=len(records))]
            samples.append(
                (
                    float(sampled["hit_chemical_accuracy"].mean()),
                    float(sampled["best_error_in_top_k"].mean()),
                    float(sampled["regret_kcal_mol"].mean()),
                )
            )
        values = np.asarray(samples)
        summaries.append(
            {
                "protocol": protocol,
                "model": model,
                "top_k": top_k,
                "hit_rate": records["hit_chemical_accuracy"].mean(),
                "hit_rate_ci_low": np.quantile(values[:, 0], 0.025),
                "hit_rate_ci_high": np.quantile(values[:, 0], 0.975),
                "mean_selected_error_kcal_mol": records["best_error_in_top_k"].mean(),
                "selected_error_ci_low": np.quantile(values[:, 1], 0.025),
                "selected_error_ci_high": np.quantile(values[:, 1], 0.975),
                "mean_regret_kcal_mol": records["regret_kcal_mol"].mean(),
                "regret_ci_low": np.quantile(values[:, 2], 0.025),
                "regret_ci_high": np.quantile(values[:, 2], 0.975),
                "evaluations_avoided_pct": records["evaluations_avoided_pct"].iloc[0],
                "n_geometries": len(records),
            }
        )
    return pd.DataFrame(summaries)


def _paired_bootstrap_differences(
    rows: pd.DataFrame,
    *,
    model: str,
    baselines: tuple[str, ...],
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    """Estimate paired top-k coverage differences against training-only baselines.

    Two intervals are reported. The geometry bootstrap resamples the 27 paired
    geometry outcomes. The molecule-cluster bootstrap first averages each
    molecule's paired outcomes and then resamples the three molecule-level
    effects, preserving dependence among geometries from the same molecule.
    """

    rng = np.random.default_rng(seed)
    protocol_rows = rows[rows["protocol"] == "geometry_grouped"]
    outputs: list[dict[str, Any]] = []
    for baseline in baselines:
        for top_k in sorted(protocol_rows["top_k"].unique()):
            selected = protocol_rows[
                (protocol_rows["top_k"] == top_k)
                & protocol_rows["model"].isin([model, baseline])
            ]
            paired = selected.pivot(
                index=["geometry_id", "molecule"],
                columns="model",
                values="hit_chemical_accuracy",
            ).dropna()
            differences = (paired[model].astype(float) - paired[baseline].astype(float)).to_numpy()
            molecule_effects = (
                paired.assign(difference=paired[model].astype(float) - paired[baseline].astype(float))
                .groupby(level="molecule")["difference"]
                .mean()
                .to_numpy()
            )
            geometry_samples = np.asarray(
                [
                    differences[rng.integers(0, len(differences), size=len(differences))].mean()
                    for _ in range(repeats)
                ]
            )
            cluster_samples = np.asarray(
                [
                    molecule_effects[
                        rng.integers(0, len(molecule_effects), size=len(molecule_effects))
                    ].mean()
                    for _ in range(repeats)
                ]
            )
            outputs.append(
                {
                    "protocol": "geometry_grouped",
                    "model": model,
                    "baseline": baseline,
                    "top_k": int(top_k),
                    "coverage_difference": float(differences.mean()),
                    "geometry_ci_low": float(np.quantile(geometry_samples, 0.025)),
                    "geometry_ci_high": float(np.quantile(geometry_samples, 0.975)),
                    "molecule_cluster_ci_low": float(np.quantile(cluster_samples, 0.025)),
                    "molecule_cluster_ci_high": float(np.quantile(cluster_samples, 0.975)),
                    "probability_difference_positive": float((geometry_samples > 0).mean()),
                    "n_geometries": len(differences),
                    "n_molecules": len(molecule_effects),
                }
            )
    return pd.DataFrame(outputs)


def run_recommender_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """Run both validation protocols and return article-ready result tables."""

    data = load_vqe_grid_dataset(config.data_root)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    grouped_predictions, grouped_baselines, grouped_params = _fit_grouped_protocol(
        data,
        config,
        protocol="geometry_grouped",
        split_groups=data["geometry_id"],
        include_molecule=True,
        outer_splits=config.outer_splits,
    )
    ablation_predictions, ablation_baselines, ablation_params = _fit_grouped_protocol(
        data,
        config,
        protocol="geometry_grouped_no_molecule",
        split_groups=data["geometry_id"],
        include_molecule=False,
        outer_splits=config.outer_splits,
    )
    molecule_predictions, molecule_baselines, molecule_params = _fit_grouped_protocol(
        data,
        config,
        protocol="leave_one_molecule_out",
        split_groups=data["molecule"],
        include_molecule=False,
        outer_splits=data["molecule"].nunique(),
    )

    predictions = pd.concat(
        [grouped_predictions, ablation_predictions, molecule_predictions], ignore_index=True
    )
    baseline_predictions = pd.concat(
        [grouped_baselines, ablation_baselines, molecule_baselines], ignore_index=True
    )
    regression = pd.concat(
        [
            _regression_metrics(grouped_predictions, "geometry_grouped"),
            _regression_metrics(ablation_predictions, "geometry_grouped_no_molecule"),
            _regression_metrics(molecule_predictions, "leave_one_molecule_out"),
        ],
        ignore_index=True,
    )
    recommendation_rows = pd.concat(
        [
            _top_k_rows(predictions),
            _top_k_rows(baseline_predictions),
            _random_baseline(data, config),
            _oracle_baseline(data, "geometry_grouped"),
            _random_baseline(data, config, protocol="geometry_grouped_no_molecule"),
            _oracle_baseline(data, "geometry_grouped_no_molecule"),
            _random_baseline(data, config, protocol="leave_one_molecule_out"),
            _oracle_baseline(data, "leave_one_molecule_out"),
        ],
        ignore_index=True,
    )
    recommendation_summary = _bootstrap_summary(
        recommendation_rows,
        repeats=config.bootstrap_repeats,
        seed=config.random_seed,
    )
    paired_differences = _paired_bootstrap_differences(
        recommendation_rows,
        model="mlp",
        baselines=("global_best", "molecule_best"),
        repeats=config.bootstrap_repeats,
        seed=config.random_seed,
    )
    by_molecule = (
        recommendation_rows.groupby(["protocol", "model", "molecule", "top_k"], as_index=False)
        .agg(
            hit_rate=("hit_chemical_accuracy", "mean"),
            mean_selected_error_kcal_mol=("best_error_in_top_k", "mean"),
            mean_regret_kcal_mol=("regret_kcal_mol", "mean"),
        )
    )

    return {
        "dataset": data,
        "predictions": predictions,
        "regression_metrics": regression,
        "recommendation_rows": recommendation_rows,
        "recommendation_summary": recommendation_summary,
        "recommendation_by_molecule": by_molecule,
        "paired_coverage_differences": paired_differences,
        "selected_hyperparameters": grouped_params + ablation_params + molecule_params,
    }
