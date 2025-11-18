"""Configuration for propensity score matching."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

MatchStrategy = Literal["nearest", "radius", "stratified"]


@dataclass
class PSMConfig:
    """Parameters for propensity score matching with reproducible settings."""

    treatment_column: str
    covariates: List[str]
    caliper: float = 0.2  # 0.2 standard deviations of propensity score logit
    match_ratio: Optional[int] = 1  # 1:1 or 1:N matching; None for greedy many
    random_state: int = 42
    strategy: MatchStrategy = "nearest"
    replacement: bool = False

    output_matched_csv: Optional[Path] = None
    output_report_txt: Optional[Path] = None

    def __post_init__(self):
        if self.match_ratio is not None and self.match_ratio < 1:
            raise ValueError(f"match_ratio must be >= 1 or None, got {self.match_ratio}")
        if self.caliper <= 0:
            raise ValueError(f"caliper must be positive, got {self.caliper}")
