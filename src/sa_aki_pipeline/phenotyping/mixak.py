"""Python interface to mixAK R clustering with model selection tracking.

This module provides utilities to run mixAK::GLMM_MCMC from Python,
track convergence diagnostics (Gelman-Rubin, autocorrelation), and
select optimal K according to the paper's criteria.

Note: Requires R installation with mixAK and coda packages.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import MixAKConfig, MixAKModelSelection, MixAKResults


def _generate_r_script(config: MixAKConfig, k: int) -> str:
    """Generate R code snippet for a single K clustering run."""
    features_r = ", ".join([f'"{f}"' for f in config.features])
    z_list = ", ".join([f'{f}=df[, "{config.time_column}"]' for f in config.features])
    x_list = ", ".join([f'{f}="empty"' for f in config.features])
    
    script = f'''
library(mixAK)
library(coda)

df <- read.csv("{config.input_csv}", as.is = TRUE)
df <- as.data.frame(df)

fea_lst <- c({features_r})
df_y <- df[, fea_lst]

k <- {len(config.features)}
ran_int_status <- TRUE
ran_eff_num <- 2 * length(k)

mod <- GLMM_MCMC(
    y = df_y,
    dist = rep("gaussian", 1),
    id = df[, "{config.id_column}"],
    x = list({x_list}),
    z = list({z_list}),
    random.intercept = rep(ran_int_status, ran_eff_num / 2),
    prior.b = list(Kmax = {k}),
    nMCMC = c(burn = {config.burn}, keep = {config.keep}, thin = {config.thin}, info = {config.info}),
    parallel = TRUE,
    PED = FALSE
)

mod <- NMixRelabel(mod, type = "stephens", keep.comp.prob = TRUE)

# Extract diagnostics
deviance <- mean(mod$Deviance)
mu_chains <- NMixChainComp(mod, relabel = TRUE, param = "mu_b")
autocorrs <- apply(mu_chains, 2, function(x) autocorr(mcmc(x), lags = 1))
autocorr_failed <- length(which(autocorrs > 0.85)) / length(autocorrs)

groupMed <- apply((mod$quant.comp.prob[["50%"]]) / 2, 1, which.max)
pHPD <- HPDinterval(mcmc(mod$comp.prob))
pHPDlower <- matrix(pHPD[, "lower"], ncol = {k}, byrow = TRUE)
pHPDupper <- matrix(pHPD[, "upper"], ncol = {k}, byrow = TRUE)
groupHPD <- groupMed
for (i in 1:{k}) {{
    groupHPD[groupHPD == i & pHPDlower[, i] <= 0.5] <- {k} + 1
}}
uncertain_count <- sum(groupHPD == {k} + 1)

# Save results
result_df <- data.frame(
    stay_id = names(groupMed),
    phenotype = groupMed,
    stringsAsFactors = FALSE
)
write.csv(result_df, "{config.output_dir}/phenotypes_k{k}.csv", row.names = FALSE)

# Save diagnostics
cat(
    sprintf("K=%d\\nDeviance=%.2f\\nAutocorr_failed=%.4f\\nUncertain=%d\\n",
            {k}, deviance, autocorr_failed, uncertain_count),
    file = "{config.output_dir}/diagnostics_k{k}.txt"
)
'''
    return script


def run_mixak_clustering(config: MixAKConfig) -> MixAKResults:
    """
    Run mixAK clustering for K=k_min..k_max and select optimal model.
    
    Parameters
    ----------
    config : MixAKConfig
        Clustering configuration with MCMC parameters and thresholds.
    
    Returns
    -------
    MixAKResults
        Selected phenotype assignments plus model selection diagnostics.
    
    Notes
    -----
    - Requires R with mixAK and coda packages installed.
    - Model selection uses Euclidean distance in (deviance, autocorr) space.
    - Gelman-Rubin diagnostic is not yet extracted (R implementation pending).
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    models: List[MixAKModelSelection] = []

    for k in range(config.k_min, config.k_max + 1):
        r_script = _generate_r_script(config, k)
        script_path = config.output_dir / f"run_k{k}.R"
        script_path.write_text(r_script, encoding="utf-8")

        try:
            subprocess.run(["Rscript", str(script_path)], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"R script for K={k} failed: {e.stderr}")
            continue

        diag_path = config.output_dir / f"diagnostics_k{k}.txt"
        if not diag_path.exists():
            continue

        diag_text = diag_path.read_text()
        deviance = float([line for line in diag_text.split("\n") if "Deviance=" in line][0].split("=")[1])
        autocorr_failed = float([line for line in diag_text.split("\n") if "Autocorr_failed=" in line][0].split("=")[1])
        uncertain = int([line for line in diag_text.split("\n") if "Uncertain=" in line][0].split("=")[1])

        models.append(
            MixAKModelSelection(
                k=k,
                deviance=deviance,
                autocorr_failed_fraction=autocorr_failed,
                uncertain_obs=uncertain,
            )
        )

    # Model selection: Euclidean distance after normalization
    selected_k = evaluate_model_selection(models, config)
    for model in models:
        model.selected = (model.k == selected_k)

    # Load phenotype assignments for selected K
    pheno_df = pd.read_csv(config.output_dir / f"phenotypes_k{selected_k}.csv")
    assignments = dict(zip(pheno_df[config.id_column], pheno_df["phenotype"]))

    output_csv = config.output_dir / f"phenotypes_k{selected_k}_final.csv"
    pheno_df.to_csv(output_csv, index=False)

    if config.save_diagnostics:
        diag_txt = config.output_dir / "model_selection_summary.txt"
        with open(diag_txt, "w", encoding="utf-8") as f:
            f.write("=== mixAK Model Selection Report ===\n\n")
            f.write(f"Selected K: {selected_k}\n\n")
            f.write("Model Comparison:\n")
            for model in sorted(models, key=lambda m: m.k):
                mark = " <-- SELECTED" if model.selected else ""
                f.write(
                    f"K={model.k}: deviance={model.deviance:.0f}, "
                    f"autocorr_failed={model.autocorr_failed_fraction:.2%}, "
                    f"uncertain={model.uncertain_obs}{mark}\n"
                )
    else:
        diag_txt = None

    return MixAKResults(
        models=models,
        selected_k=selected_k,
        phenotype_assignments=assignments,
        output_csv=output_csv,
        diagnostics_txt=diag_txt,
    )


def evaluate_model_selection(models: List[MixAKModelSelection], config: MixAKConfig) -> int:
    """
    Select optimal K using normalized Euclidean distance.
    
    Criteria (from original notebooks):
    - Minimize deviance
    - Minimize fraction of chains with high autocorrelation (>0.85 in code, >0.1 in paper)
    - Gelman-Rubin < 1.1 (not yet enforced due to R extraction complexity)
    
    Returns
    -------
    int
        Selected number of clusters.
    """
    if not models:
        raise ValueError("No valid models to select from")

    deviances = np.array([m.deviance for m in models])
    autocorrs = np.array([m.autocorr_failed_fraction for m in models])

    # Normalize to [0, 1] where 0=best, 1=worst
    def scale(arr):
        """Normalize so min value -> 0, max value -> 1."""
        if arr.max() == arr.min():
            return np.zeros_like(arr)
        return (arr - arr.min()) / (arr.max() - arr.min())

    scaled_dev = scale(deviances)
    scaled_autocorr = scale(autocorrs)

    # Euclidean distance from origin (0,0) - lower is better
    scores = np.sqrt(scaled_dev**2 + scaled_autocorr**2)
    best_idx = np.argmin(scores)

    return models[best_idx].k
