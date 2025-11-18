#!/usr/bin/env python3
"""Analyze sepsis→AKI timing intervals by phenotype.

Usage:
    python run_sepsis_aki_timing.py \\
        --event-times event_times.csv \\
        --phenotypes phenotypes.csv \\
        --output-csv intervals.csv \\
        --output-plot intervals.pdf
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from sa_aki_pipeline.sepsis import (
    SepsisAKITimingConfig,
    calculate_sepsis_aki_interval,
    plot_sepsis_aki_boxplot
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze sepsis→AKI timing intervals'
    )
    parser.add_argument(
        '--event-times',
        required=True,
        help='CSV with sepsis_onset and saki_onset columns'
    )
    parser.add_argument(
        '--phenotypes',
        required=True,
        help='CSV with stay_id and groupHPD columns'
    )
    parser.add_argument(
        '--output-csv',
        required=True,
        help='Output path for interval statistics'
    )
    parser.add_argument(
        '--output-plot',
        help='Optional output path for boxplot figure'
    )
    parser.add_argument(
        '--comparison-test',
        default='t-test_welch',
        choices=['t-test_welch', 't-test_ind', 'Mann-Whitney'],
        help='Statistical test for comparisons (default: t-test_welch)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Sepsis→AKI Timing Analysis")
    logger.info("=" * 60)
    logger.info(f"Event times: {args.event_times}")
    logger.info(f"Phenotypes: {args.phenotypes}")
    logger.info(f"Comparison test: {args.comparison_test}")
    logger.info("")
    
    # Load data
    event_times = pd.read_csv(args.event_times)
    phenotypes = pd.read_csv(args.phenotypes)
    
    logger.info(f"Loaded {len(event_times)} event time records")
    logger.info(f"Loaded {len(phenotypes)} phenotype assignments")
    
    # Create config
    config = SepsisAKITimingConfig(
        comparison_test=args.comparison_test
    )
    
    # Calculate intervals
    result = calculate_sepsis_aki_interval(
        event_times_df=event_times,
        phenotype_df=phenotypes,
        config=config
    )
    
    # Save results
    result.timing_data.to_csv(args.output_csv, index=False)
    logger.info(f"\n✓ Interval data saved to: {args.output_csv}")
    
    # Save summary statistics
    summary_path = args.output_csv.replace('.csv', '_summary.csv')
    result.summary_stats.to_csv(summary_path)
    logger.info(f"✓ Summary statistics saved to: {summary_path}")
    
    # Save pairwise tests
    pairwise_path = args.output_csv.replace('.csv', '_pairwise.csv')
    result.pairwise_tests.to_csv(pairwise_path, index=False)
    logger.info(f"✓ Pairwise tests saved to: {pairwise_path}")
    
    # Create plot if requested
    if args.output_plot:
        plot_sepsis_aki_boxplot(result, args.output_plot)
        logger.info(f"✓ Boxplot saved to: {args.output_plot}")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary Statistics")
    logger.info("=" * 60)
    print(result.summary_stats)
    
    logger.info("\n" + "=" * 60)
    logger.info("Pairwise Comparisons")
    logger.info("=" * 60)
    print(result.pairwise_tests)
    
    logger.info("\n✓ Analysis complete!")


if __name__ == '__main__':
    main()
