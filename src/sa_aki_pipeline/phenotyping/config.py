"""Configuration for mixAK clustering workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class MixAKConfig:
    """Parameters for mixAK longitudinal clustering in R.
    
    These parameters match the original notebook workflow and ensure
    reproducibility for Methods section citation.
    """

    # Data specification
    input_csv: Path
    id_column: str = "stay_id"
    time_column: str = "time"
    features: List[str] = field(default_factory=list)

    # Clustering range
    k_min: int = 2
    k_max: int = 5

    # MCMC parameters (as used in original notebooks)
    burn: int = 50
    keep: int = 2000  # reduced to 500 for K>=4 in original code
    thin: int = 50
    info: int = 50

    # Model selection thresholds
    autocorr_threshold: float = 0.1  # <10% chains with autocorr > threshold
    gelman_rubin_threshold: float = 1.1  # R-hat < 1.1 for convergence

    # Output paths
    output_dir: Path = Path("outputs/mixak")
    save_diagnostics: bool = True
    r_script_template: Optional[Path] = None


@dataclass
class MixAKModelSelection:
    """Results from mixAK model selection procedure."""

    k: int
    deviance: float
    autocorr_failed_fraction: float
    uncertain_obs: int
    gelman_rubin_max: Optional[float] = None
    selected: bool = False

    def quality_score(self) -> float:
        """Compute normalized Euclidean distance for model ranking."""
        import numpy as np
        # Placeholder; requires normalization across all K
        return np.sqrt(self.deviance**2 + self.autocorr_failed_fraction**2)


@dataclass
class MixAKResults:
    """Container for all clustering results."""

    models: List[MixAKModelSelection]
    selected_k: int
    phenotype_assignments: Dict[str, int]  # {stay_id: phenotype}
    output_csv: Path
    diagnostics_txt: Optional[Path] = None

    def summary_table(self) -> str:
        """Generate Methods-ready summary table."""
        lines = ["K\tDeviance\tAutocorr>0.1\tUncertain\tSelected"]
        for model in sorted(self.models, key=lambda m: m.k):
            mark = "*" if model.k == self.selected_k else ""
            lines.append(
                f"{model.k}\t{model.deviance:.0f}\t"
                f"{model.autocorr_failed_fraction:.2%}\t"
                f"{model.uncertain_obs}\t{mark}"
            )
        return "\n".join(lines)
