"""Propensity score matching with standardized diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from .config import PSMConfig


@dataclass
class PSMResult:
    """Return value encapsulating matched cohort and diagnostics."""

    matched_df: pd.DataFrame
    propensity_scores: pd.Series
    n_treated: int
    n_control: int
    n_matched_treated: int
    n_matched_control: int
    smd_before: Dict[str, float]
    smd_after: Dict[str, float]
    config: PSMConfig


def compute_standardized_mean_difference(
    df: pd.DataFrame,
    treatment_col: str,
    covariates: List[str],
) -> Dict[str, float]:
    """Compute SMD for each covariate between treatment=1 and treatment=0."""
    treated = df[df[treatment_col] == 1][covariates]
    control = df[df[treatment_col] == 0][covariates]
    smd = {}
    for col in covariates:
        mean_t = treated[col].mean()
        mean_c = control[col].mean()
        var_t = treated[col].var()
        var_c = control[col].var()
        pooled_std = np.sqrt((var_t + var_c) / 2)
        if pooled_std == 0:
            smd[col] = 0.0
        else:
            smd[col] = (mean_t - mean_c) / pooled_std
    return smd


def _fit_propensity_model(
    df: pd.DataFrame,
    treatment_col: str,
    covariates: List[str],
    random_state: int,
) -> Tuple[np.ndarray, LogisticRegression]:
    X = df[covariates].fillna(df[covariates].median())
    y = df[treatment_col].values
    model = LogisticRegression(random_state=random_state, max_iter=1000, solver="lbfgs")
    model.fit(X, y)
    ps = model.predict_proba(X)[:, 1]
    return ps, model


def _nearest_neighbor_match(
    df: pd.DataFrame,
    treatment_col: str,
    propensity_scores: np.ndarray,
    caliper_std: float,
    match_ratio: Optional[int],
    replacement: bool,
    random_state: int,
) -> pd.DataFrame:
    logit_ps = np.log(propensity_scores / (1 - propensity_scores + 1e-9))
    caliper_distance = caliper_std * logit_ps.std()

    treated_idx = df[df[treatment_col] == 1].index.values
    control_idx = df[df[treatment_col] == 0].index.values

    if len(control_idx) == 0 or len(treated_idx) == 0:
        return pd.DataFrame()

    control_ps = logit_ps[control_idx].reshape(-1, 1)
    treated_ps = logit_ps[treated_idx].reshape(-1, 1)

    n_neighbors = match_ratio if match_ratio is not None else len(control_idx)
    nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(control_idx)), algorithm="ball_tree")
    nn.fit(control_ps)

    matched_pairs = []
    used_control = set()
    np.random.seed(random_state)
    shuffle_order = np.random.permutation(len(treated_idx))

    for i in shuffle_order:
        t_idx = treated_idx[i]
        distances, indices = nn.kneighbors(treated_ps[i].reshape(1, -1))
        for dist, candidate_pos in zip(distances[0], indices[0]):
            c_idx = control_idx[candidate_pos]
            if dist > caliper_distance:
                continue
            if not replacement and c_idx in used_control:
                continue
            matched_pairs.append({"treated_idx": t_idx, "control_idx": c_idx, "distance": dist})
            if not replacement:
                used_control.add(c_idx)
            if match_ratio and len([p for p in matched_pairs if p["treated_idx"] == t_idx]) >= match_ratio:
                break

    if not matched_pairs:
        return pd.DataFrame()

    matched_indices = set()
    for pair in matched_pairs:
        matched_indices.add(pair["treated_idx"])
        matched_indices.add(pair["control_idx"])

    return df.loc[list(matched_indices)].copy()


def propensity_score_matching(df: pd.DataFrame, config: PSMConfig) -> PSMResult:
    """
    Perform propensity score matching with reproducible parameters.

    Parameters
    ----------
    df : pd.DataFrame
        Input cohort with treatment and covariates.
    config : PSMConfig
        Matching configuration including caliper=0.2*SD(logit), match_ratio, etc.

    Returns
    -------
    PSMResult
        Matched dataframe plus diagnostics (SMD before/after, sample sizes).

    Notes
    -----
    - Caliper is applied to the logit of the propensity score.
    - SMD (standardized mean difference) is computed for all covariates.
    - Results can be serialized to CSV and text report for Methods section citation.
    """
    smd_before = compute_standardized_mean_difference(df, config.treatment_column, config.covariates)
    ps, model = _fit_propensity_model(df, config.treatment_column, config.covariates, config.random_state)

    if config.strategy == "nearest":
        matched = _nearest_neighbor_match(
            df,
            config.treatment_column,
            ps,
            config.caliper,
            config.match_ratio,
            config.replacement,
            config.random_state,
        )
    else:
        raise NotImplementedError(f"Strategy '{config.strategy}' not yet implemented; use 'nearest'.")

    if matched.empty:
        smd_after = {k: np.nan for k in smd_before}
    else:
        smd_after = compute_standardized_mean_difference(matched, config.treatment_column, config.covariates)

    n_treated = int((df[config.treatment_column] == 1).sum())
    n_control = int((df[config.treatment_column] == 0).sum())
    n_matched_treated = int((matched[config.treatment_column] == 1).sum()) if not matched.empty else 0
    n_matched_control = int((matched[config.treatment_column] == 0).sum()) if not matched.empty else 0

    result = PSMResult(
        matched_df=matched,
        propensity_scores=pd.Series(ps, index=df.index),
        n_treated=n_treated,
        n_control=n_control,
        n_matched_treated=n_matched_treated,
        n_matched_control=n_matched_control,
        smd_before=smd_before,
        smd_after=smd_after,
        config=config,
    )

    if config.output_matched_csv and not matched.empty:
        Path(config.output_matched_csv).parent.mkdir(parents=True, exist_ok=True)
        matched.to_csv(config.output_matched_csv, index=False)

    if config.output_report_txt:
        _write_report(result, config.output_report_txt)

    return result


def _write_report(result: PSMResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== Propensity Score Matching Report ===\n\n")
        f.write(f"Treatment column: {result.config.treatment_column}\n")
        f.write(f"Covariates: {', '.join(result.config.covariates)}\n")
        f.write(f"Caliper: {result.config.caliper} SD of logit propensity score\n")
        f.write(f"Match ratio: {result.config.match_ratio if result.config.match_ratio else 'greedy'}\n")
        f.write(f"Replacement: {result.config.replacement}\n")
        f.write(f"Random state: {result.config.random_state}\n\n")
        f.write(f"Original cohort: {result.n_treated} treated, {result.n_control} control\n")
        f.write(f"Matched cohort: {result.n_matched_treated} treated, {result.n_matched_control} control\n\n")
        f.write("Standardized Mean Differences (SMD):\n")
        f.write(f"{'Covariate':<30} {'Before':<12} {'After':<12}\n")
        f.write("-" * 60 + "\n")
        for cov in result.config.covariates:
            before = result.smd_before.get(cov, np.nan)
            after = result.smd_after.get(cov, np.nan)
            f.write(f"{cov:<30} {before:>11.4f} {after:>11.4f}\n")
