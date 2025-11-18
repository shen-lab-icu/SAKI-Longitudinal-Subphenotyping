"""Configuration for sepsis-induced AKI onset analysis."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class SepsisAKITimingConfig:
    """Configuration for sepsis→AKI timing analysis.
    
    Methods Citation:
        Time intervals calculated between sepsis onset (Sepsis-3 criteria) 
        and SAKI onset (KDIGO criteria). Intervals stratified by phenotype
        and compared using Welch's t-test.
    """
    
    # Time fields
    sepsis_onset_field: str = 'sepsis_onset'
    saki_onset_field: str = 'saki_onset'
    
    # Derived interval field
    interval_field: str = 'los_saki-sepsis'
    
    # Statistical test
    comparison_test: str = 't-test_welch'  # Options: t-test_ind, Mann-Whitney, Wilcoxon
    
    # Significance level
    alpha: float = 0.05


@dataclass
class SOFAEvolutionConfig:
    """Configuration for SOFA score evolution around sepsis/AKI onset.
    
    Methods Citation:
        SOFA scores extracted at sepsis onset and AKI onset timepoints.
        Time windows: 12h before sepsis to sepsis onset, and sepsis onset to 12h after AKI.
        Compared using Welch's t-test within phenotypes.
    """
    
    # Time windows (hours relative to event)
    sepsis_window_start: int = -12  # 12h before sepsis
    sepsis_window_end: int = 0      # At sepsis onset
    
    saki_window_start: int = 0      # At sepsis onset (= SAKI baseline)
    saki_window_end: int = 12       # 12h after SAKI onset
    
    # Features to track
    features: List[str] = field(default_factory=lambda: [
        'sofa',
        'sofa_norenal',
        'mbp',
        'lactate',
        'creatinine',
        'urineoutput'
    ])
    
    # Comparison settings
    comparison_test: str = 't-test_welch'
    paired_comparison: bool = True  # Compare sepsis vs SAKI within same patient
    
    # Statistical parameters
    alpha: float = 0.05
    multiple_testing_correction: str = 'bonferroni'  # None, bonferroni, fdr_bh


@dataclass
class InfectionTrajectoryConfig:
    """Configuration for infection feature trajectory analysis.
    
    Methods Citation:
        Infection-related features tracked longitudinally across phenotypes.
        Features include culture positivity, antibiotic coverage, infection site.
    """
    
    # Infection features
    culture_positive_field: str = 'culture_positive'
    antibiotic_coverage_field: str = 'abx_coverage'
    infection_site_field: str = 'infection_site'
    
    # Site categories
    infection_sites: List[str] = field(default_factory=lambda: [
        'respiratory',
        'urinary',
        'abdominal',
        'bloodstream',
        'other'
    ])
    
    # Trajectory time windows (hours from sepsis onset)
    trajectory_windows: List[int] = field(default_factory=lambda: [
        -24, -12, 0, 12, 24, 48, 72
    ])
