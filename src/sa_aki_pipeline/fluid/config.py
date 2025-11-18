"""Configuration for fluid resuscitation and diuretic response analysis."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DiureticResponseConfig:
    """Configuration for diuretic response PSM analysis.
    
    Implements three-way propensity score matching (TriMatch) for comparing:
    - No diuretic administration
    - Non-responsive to diuretics
    - Responsive to diuretics
    
    Methods Citation:
        Three-way PSM was performed using TriMatch with caliper ranges:
        - Phenotype 1: caliper=0.05
        - Phenotype 2: M1=1.5, M2=4 (OneToN matching)
        - Phenotype 3: caliper=0.14
        
        Matching variables: creatinine, urine output, SOFA (non-renal), colloid bolus
    """
    
    # Matching variables (used across all phenotypes)
    match_vars: List[str] = field(default_factory=lambda: [
        'creatinine', 
        'urineoutput', 
        'sofa_norenal', 
        'colloid_bolus'
    ])
    
    # Phenotype-specific matching parameters
    phenotype1_caliper: float = 0.05
    phenotype2_M1: float = 1.5  # Max times treat1 can be used
    phenotype2_M2: int = 4      # Max times treat1 matched with treat2
    phenotype3_caliper: float = 0.14
    
    # Response classification field
    response_field: str = 'label_diu_res'
    
    # Expected response categories
    response_categories: List[str] = field(default_factory=lambda: [
        'No diuretic',
        'Non-responsive', 
        'Responsive'
    ])


@dataclass
class FluidBalanceConfig:
    """Configuration for fluid input/output balance calculation.
    
    Methods Citation:
        Fluid balance calculated as cumulative (input - output) over 
        specified time windows, with colloid/crystalloid stratification.
    """
    
    # Time window for balance calculation (hours)
    window_hours: int = 24
    
    # Fluid types to track
    colloid_types: List[str] = field(default_factory=lambda: [
        'albumin',
        'hetastarch', 
        'dextran'
    ])
    
    crystalloid_types: List[str] = field(default_factory=lambda: [
        'normal_saline',
        'lactated_ringers',
        'plasmalyte'
    ])
    
    # Output tracking
    urine_output_field: str = 'urineoutput'
    
    
@dataclass
class DiureticDoseConfig:
    """Configuration for diuretic dose analysis.
    
    Methods Citation:
        Diuretic doses standardized to furosemide-equivalent units.
        Response defined as urine output ≥100 mL/hr within 6h post-dose.
    """
    
    # Furosemide equivalents (relative potency)
    furosemide_equiv: dict = field(default_factory=lambda: {
        'furosemide': 1.0,
        'bumetanide': 40.0,  # 1 mg bumetanide = 40 mg furosemide
        'torsemide': 2.0      # 1 mg torsemide = 2 mg furosemide
    })
    
    # Response criteria
    response_window_hours: int = 6
    response_uo_threshold_ml_hr: float = 100.0
    
    # Dose bins for stratification
    dose_bins_mg: List[float] = field(default_factory=lambda: [
        0, 40, 80, 120, 160, float('inf')
    ])
    
    dose_labels: List[str] = field(default_factory=lambda: [
        '0-40mg',
        '40-80mg', 
        '80-120mg',
        '120-160mg',
        '>160mg'
    ])
