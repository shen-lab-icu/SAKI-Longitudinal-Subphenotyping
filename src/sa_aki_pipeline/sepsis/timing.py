"""Sepsis-induced AKI timing and SOFA evolution analysis.

This module implements analyses from:
05.sepsis_induced_AKI/sofa in sepsis and AKI onset-v2.ipynb

Key features:
- Time interval calculation between sepsis and SAKI onset
- SOFA score evolution at key timepoints
- Feature comparison between onset events
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

from .config import SepsisAKITimingConfig, SOFAEvolutionConfig

logger = logging.getLogger(__name__)


@dataclass
class SepsisAKITimingResult:
    """Results from sepsis→AKI timing analysis."""
    
    timing_data: pd.DataFrame  # stay_id, phenotype, interval
    summary_stats: pd.DataFrame  # Per-phenotype mean, std, median
    pairwise_tests: pd.DataFrame  # Statistical test results between phenotypes
    config: SepsisAKITimingConfig


@dataclass
class SOFAEvolutionResult:
    """Results from SOFA evolution analysis."""
    
    sepsis_onset_features: pd.DataFrame  # Features at sepsis onset
    saki_onset_features: pd.DataFrame    # Features at AKI onset
    paired_comparisons: pd.DataFrame     # Within-patient differences
    phenotype_comparisons: pd.DataFrame  # Between-phenotype tests
    config: SOFAEvolutionConfig


def calculate_sepsis_aki_interval(
    event_times_df: pd.DataFrame,
    phenotype_df: pd.DataFrame,
    config: Optional[SepsisAKITimingConfig] = None
) -> SepsisAKITimingResult:
    """Calculate time intervals between sepsis and SAKI onset.
    
    Args:
        event_times_df: DataFrame with sepsis_onset and saki_onset columns
        phenotype_df: DataFrame with stay_id and groupHPD columns
        config: Timing analysis configuration
        
    Returns:
        SepsisAKITimingResult with interval statistics
    """
    if config is None:
        config = SepsisAKITimingConfig()
    
    # Merge phenotype assignments
    df = pd.merge(
        phenotype_df[['stay_id', 'groupHPD']],
        event_times_df[['stay_id', config.sepsis_onset_field, config.saki_onset_field]],
        on='stay_id',
        how='inner'
    )
    
    # Calculate interval (in hours)
    df[config.interval_field] = (
        pd.to_datetime(df[config.saki_onset_field]) - 
        pd.to_datetime(df[config.sepsis_onset_field])
    ).dt.total_seconds() / 3600
    
    logger.info(f"Calculated intervals for {len(df)} patients")
    logger.info(f"Interval range: {df[config.interval_field].min():.1f} to {df[config.interval_field].max():.1f} hours")
    
    # Summary statistics by phenotype
    summary_stats = df.groupby('groupHPD')[config.interval_field].agg([
        'count', 'mean', 'std', 'median',
        ('q25', lambda x: x.quantile(0.25)),
        ('q75', lambda x: x.quantile(0.75))
    ]).round(2)
    
    logger.info(f"\nSummary by phenotype:\n{summary_stats}")
    
    # Pairwise statistical tests
    from scipy import stats
    
    phenotypes = sorted(df['groupHPD'].unique())
    pairwise_results = []
    
    for i, p1 in enumerate(phenotypes):
        for p2 in phenotypes[i+1:]:
            data1 = df[df['groupHPD'] == p1][config.interval_field].dropna()
            data2 = df[df['groupHPD'] == p2][config.interval_field].dropna()
            
            if config.comparison_test == 't-test_welch':
                stat, pval = stats.ttest_ind(data1, data2, equal_var=False)
            elif config.comparison_test == 'Mann-Whitney':
                stat, pval = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            else:
                stat, pval = stats.ttest_ind(data1, data2)
            
            pairwise_results.append({
                'phenotype1': p1,
                'phenotype2': p2,
                'mean_diff': data1.mean() - data2.mean(),
                'statistic': stat,
                'p_value': pval,
                'significant': pval < config.alpha
            })
    
    pairwise_df = pd.DataFrame(pairwise_results)
    
    logger.info(f"\nPairwise comparisons:\n{pairwise_df}")
    
    return SepsisAKITimingResult(
        timing_data=df,
        summary_stats=summary_stats,
        pairwise_tests=pairwise_df,
        config=config
    )


def extract_sofa_at_timepoints(
    longitudinal_df: pd.DataFrame,
    event_times_df: pd.DataFrame,
    phenotype_df: pd.DataFrame,
    config: Optional[SOFAEvolutionConfig] = None
) -> SOFAEvolutionResult:
    """Extract SOFA and related features at sepsis/AKI onset timepoints.
    
    Args:
        longitudinal_df: Time-series data with timestamps and features
        event_times_df: DataFrame with event onset times
        phenotype_df: DataFrame with phenotype assignments
        config: SOFA evolution configuration
        
    Returns:
        SOFAEvolutionResult with onset-aligned features
    """
    if config is None:
        config = SOFAEvolutionConfig()
    
    # Merge phenotypes and event times
    df_base = pd.merge(
        phenotype_df[['stay_id', 'groupHPD']],
        event_times_df[['stay_id', 'sepsis_onset', 'saki_onset']],
        on='stay_id',
        how='inner'
    )
    
    # Convert timestamps
    df_base['sepsis_onset'] = pd.to_datetime(df_base['sepsis_onset'])
    df_base['saki_onset'] = pd.to_datetime(df_base['saki_onset'])
    
    longitudinal_df = longitudinal_df.copy()
    longitudinal_df['timestamp'] = pd.to_datetime(longitudinal_df['timestamp'])
    
    logger.info(f"Extracting features for {len(df_base)} patients")
    
    # Extract features at sepsis onset
    sepsis_features = []
    saki_features = []
    
    for _, row in df_base.iterrows():
        stay_id = row['stay_id']
        sepsis_time = row['sepsis_onset']
        saki_time = row['saki_onset']
        
        patient_data = longitudinal_df[longitudinal_df['stay_id'] == stay_id].copy()
        
        if len(patient_data) == 0:
            continue
        
        # Features at sepsis onset window
        sepsis_window = patient_data[
            (patient_data['timestamp'] >= sepsis_time + pd.Timedelta(hours=config.sepsis_window_start)) &
            (patient_data['timestamp'] <= sepsis_time + pd.Timedelta(hours=config.sepsis_window_end))
        ]
        
        if len(sepsis_window) > 0:
            sepsis_vals = {
                'stay_id': stay_id,
                'groupHPD': row['groupHPD'],
                'status': 'Sepsis'
            }
            for feat in config.features:
                if feat in sepsis_window.columns:
                    sepsis_vals[feat] = sepsis_window[feat].mean()
            sepsis_features.append(sepsis_vals)
        
        # Features at SAKI onset window
        saki_window = patient_data[
            (patient_data['timestamp'] >= saki_time + pd.Timedelta(hours=config.saki_window_start)) &
            (patient_data['timestamp'] <= saki_time + pd.Timedelta(hours=config.saki_window_end))
        ]
        
        if len(saki_window) > 0:
            saki_vals = {
                'stay_id': stay_id,
                'groupHPD': row['groupHPD'],
                'status': 'S-AKI'
            }
            for feat in config.features:
                if feat in saki_window.columns:
                    saki_vals[feat] = saki_window[feat].mean()
            saki_features.append(saki_vals)
    
    df_sepsis = pd.DataFrame(sepsis_features)
    df_saki = pd.DataFrame(saki_features)
    
    logger.info(f"Extracted sepsis onset features: {len(df_sepsis)} samples")
    logger.info(f"Extracted SAKI onset features: {len(df_saki)} samples")
    
    # Paired comparisons (within-patient)
    paired_data = pd.merge(
        df_sepsis,
        df_saki,
        on=['stay_id', 'groupHPD'],
        suffixes=('_sepsis', '_saki')
    )
    
    paired_comparisons = []
    
    for feat in config.features:
        sepsis_col = f'{feat}_sepsis'
        saki_col = f'{feat}_saki'
        
        if sepsis_col not in paired_data.columns or saki_col not in paired_data.columns:
            continue
        
        for phenotype in sorted(paired_data['groupHPD'].unique()):
            pheno_data = paired_data[paired_data['groupHPD'] == phenotype]
            
            sepsis_vals = pheno_data[sepsis_col].dropna()
            saki_vals = pheno_data[saki_col].dropna()
            
            if len(sepsis_vals) < 5 or len(saki_vals) < 5:
                continue
            
            from scipy import stats
            
            if config.paired_comparison:
                # Paired t-test
                valid_pairs = pheno_data[[sepsis_col, saki_col]].dropna()
                if len(valid_pairs) >= 5:
                    stat, pval = stats.ttest_rel(
                        valid_pairs[sepsis_col],
                        valid_pairs[saki_col]
                    )
                else:
                    stat, pval = np.nan, np.nan
            else:
                # Independent t-test
                stat, pval = stats.ttest_ind(sepsis_vals, saki_vals, equal_var=False)
            
            paired_comparisons.append({
                'feature': feat,
                'phenotype': phenotype,
                'sepsis_mean': sepsis_vals.mean(),
                'sepsis_std': sepsis_vals.std(),
                'saki_mean': saki_vals.mean(),
                'saki_std': saki_vals.std(),
                'mean_diff': saki_vals.mean() - sepsis_vals.mean(),
                'statistic': stat,
                'p_value': pval,
                'n_pairs': len(pheno_data)
            })
    
    paired_comp_df = pd.DataFrame(paired_comparisons)
    
    # Apply multiple testing correction if requested
    if config.multiple_testing_correction and len(paired_comp_df) > 0:
        from scipy.stats import false_discovery_control
        
        if config.multiple_testing_correction == 'bonferroni':
            paired_comp_df['p_value_adj'] = paired_comp_df['p_value'] * len(paired_comp_df)
            paired_comp_df['p_value_adj'] = paired_comp_df['p_value_adj'].clip(upper=1.0)
        elif config.multiple_testing_correction == 'fdr_bh':
            paired_comp_df['p_value_adj'] = false_discovery_control(
                paired_comp_df['p_value'].fillna(1.0)
            )
        
        paired_comp_df['significant'] = paired_comp_df['p_value_adj'] < config.alpha
    else:
        paired_comp_df['significant'] = paired_comp_df['p_value'] < config.alpha
    
    logger.info(f"\nPaired comparisons completed: {len(paired_comp_df)} tests")
    logger.info(f"Significant changes: {paired_comp_df['significant'].sum()}")
    
    # Between-phenotype comparisons (at each timepoint)
    phenotype_comparisons = []
    
    for status, df_status in [('Sepsis', df_sepsis), ('S-AKI', df_saki)]:
        for feat in config.features:
            if feat not in df_status.columns:
                continue
            
            phenotypes = sorted(df_status['groupHPD'].unique())
            
            for i, p1 in enumerate(phenotypes):
                for p2 in phenotypes[i+1:]:
                    data1 = df_status[df_status['groupHPD'] == p1][feat].dropna()
                    data2 = df_status[df_status['groupHPD'] == p2][feat].dropna()
                    
                    if len(data1) < 5 or len(data2) < 5:
                        continue
                    
                    from scipy import stats
                    stat, pval = stats.ttest_ind(data1, data2, equal_var=False)
                    
                    phenotype_comparisons.append({
                        'timepoint': status,
                        'feature': feat,
                        'phenotype1': p1,
                        'phenotype2': p2,
                        'mean1': data1.mean(),
                        'mean2': data2.mean(),
                        'mean_diff': data1.mean() - data2.mean(),
                        'statistic': stat,
                        'p_value': pval,
                        'significant': pval < config.alpha
                    })
    
    phenotype_comp_df = pd.DataFrame(phenotype_comparisons)
    
    logger.info(f"Phenotype comparisons: {len(phenotype_comp_df)} tests")
    
    return SOFAEvolutionResult(
        sepsis_onset_features=df_sepsis,
        saki_onset_features=df_saki,
        paired_comparisons=paired_comp_df,
        phenotype_comparisons=phenotype_comp_df,
        config=config
    )


def plot_sepsis_aki_boxplot(
    result: SepsisAKITimingResult,
    output_path: Optional[str] = None
) -> None:
    """Create boxplot of sepsis→AKI intervals by phenotype.
    
    Args:
        result: SepsisAKITimingResult from calculate_sepsis_aki_interval
        output_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    df = result.timing_data
    
    fig, ax = plt.subplots(figsize=(4, 3))
    
    sns.boxplot(
        data=df,
        x='groupHPD',
        y=result.config.interval_field,
        palette='Set3',
        showfliers=False,
        showmeans=True,
        meanprops={'marker': 'D', 'markerfacecolor': 'white'},
        whis=0.5,
        ax=ax
    )
    
    # Add statistical annotations
    try:
        from statannotations.Annotator import Annotator
        
        phenotypes = sorted(df['groupHPD'].unique())
        pairs = [(p1, p2) for i, p1 in enumerate(phenotypes) 
                 for p2 in phenotypes[i+1:]]
        
        annotator = Annotator(
            ax,
            pairs=pairs,
            data=df,
            x='groupHPD',
            y=result.config.interval_field
        )
        annotator.configure(
            test=result.config.comparison_test,
            text_format='star',
            line_height=0.03,
            line_width=1,
            loc='outside'
        )
        annotator.apply_and_annotate()
    except ImportError:
        logger.warning("statannotations not available, skipping significance stars")
    
    ax.set_xlabel('Phenotype')
    ax.set_ylabel('Sepsis → SAKI interval (hours)')
    ax.set_title('Time to AKI onset by phenotype')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {output_path}")
    
    plt.show()


def plot_sofa_evolution_boxplot(
    result: SOFAEvolutionResult,
    feature: str,
    output_path: Optional[str] = None
) -> None:
    """Create paired boxplot of feature at sepsis vs SAKI onset.
    
    Args:
        result: SOFAEvolutionResult from extract_sofa_at_timepoints
        feature: Feature name to plot
        output_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Combine sepsis and SAKI data
    df_sepsis = result.sepsis_onset_features.copy()
    df_saki = result.saki_onset_features.copy()
    
    if feature not in df_sepsis.columns or feature not in df_saki.columns:
        raise ValueError(f"Feature '{feature}' not found in results")
    
    df_combined = pd.concat([df_sepsis, df_saki], ignore_index=True)
    
    # Map phenotypes to labels
    df_combined['groupHPD'] = df_combined['groupHPD'].map({
        1: 'C1', 2: 'C2', 3: 'C3'
    })
    
    fig, ax = plt.subplots(figsize=(4, 3))
    
    sns.boxplot(
        data=df_combined,
        x='groupHPD',
        y=feature,
        hue='status',
        order=['C1', 'C2', 'C3'],
        palette='Set2',
        saturation=0.4,
        showfliers=False,
        showmeans=True,
        ax=ax
    )
    
    # Add statistical annotations
    try:
        from statannotations.Annotator import Annotator
        
        pairs = [
            (('C1', 'Sepsis'), ('C1', 'S-AKI')),
            (('C2', 'Sepsis'), ('C2', 'S-AKI')),
            (('C3', 'Sepsis'), ('C3', 'S-AKI'))
        ]
        
        annotator = Annotator(
            ax,
            pairs=pairs,
            data=df_combined,
            x='groupHPD',
            y=feature,
            hue='status',
            order=['C1', 'C2', 'C3']
        )
        annotator.configure(
            test=result.config.comparison_test,
            text_format='star',
            loc='outside'
        )
        annotator.apply_and_annotate()
    except ImportError:
        logger.warning("statannotations not available, skipping significance stars")
    
    ax.set_xlabel('Phenotype')
    ax.set_ylabel(feature)
    ax.set_title(f'{feature} at sepsis vs. AKI onset')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {output_path}")
    
    plt.show()
