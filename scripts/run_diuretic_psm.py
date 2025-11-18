#!/usr/bin/env python3
"""Run three-way diuretic response PSM analysis.

Usage:
    python run_diuretic_psm.py --input-csv diuretic_data.csv --output-csv matched.csv
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from sa_aki_pipeline.fluid import (
    DiureticResponseConfig,
    run_diuretic_psm_r,
    run_diuretic_psm_python
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Three-way PSM for diuretic response analysis'
    )
    parser.add_argument(
        '--input-csv',
        required=True,
        help='Input CSV with diuretic response data'
    )
    parser.add_argument(
        '--output-csv',
        required=True,
        help='Output path for matched cohort'
    )
    parser.add_argument(
        '--use-r',
        action='store_true',
        help='Use R TriMatch (recommended). If False, uses Python approximation.'
    )
    parser.add_argument(
        '--phenotype1-caliper',
        type=float,
        default=0.05,
        help='Caliper for phenotype 1 (default: 0.05)'
    )
    parser.add_argument(
        '--phenotype2-M1',
        type=float,
        default=1.5,
        help='M1 parameter for phenotype 2 OneToN matching (default: 1.5)'
    )
    parser.add_argument(
        '--phenotype2-M2',
        type=int,
        default=4,
        help='M2 parameter for phenotype 2 OneToN matching (default: 4)'
    )
    parser.add_argument(
        '--phenotype3-caliper',
        type=float,
        default=0.14,
        help='Caliper for phenotype 3 (default: 0.14)'
    )
    parser.add_argument(
        '--response-field',
        default='label_diu_res',
        help='Column name for response categories (default: label_diu_res)'
    )
    
    args = parser.parse_args()
    
    # Create config
    config = DiureticResponseConfig(
        phenotype1_caliper=args.phenotype1_caliper,
        phenotype2_M1=args.phenotype2_M1,
        phenotype2_M2=args.phenotype2_M2,
        phenotype3_caliper=args.phenotype3_caliper,
        response_field=args.response_field
    )
    
    logger.info("=" * 60)
    logger.info("Three-way Diuretic Response PSM")
    logger.info("=" * 60)
    logger.info(f"Input: {args.input_csv}")
    logger.info(f"Output: {args.output_csv}")
    logger.info(f"Method: {'R TriMatch' if args.use_r else 'Python (approximation)'}")
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  Phenotype 1 caliper: {config.phenotype1_caliper}")
    logger.info(f"  Phenotype 2 M1={config.phenotype2_M1}, M2={config.phenotype2_M2}")
    logger.info(f"  Phenotype 3 caliper: {config.phenotype3_caliper}")
    logger.info(f"  Matching variables: {', '.join(config.match_vars)}")
    logger.info("")
    
    # Run PSM
    if args.use_r:
        result = run_diuretic_psm_r(
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            config=config
        )
    else:
        logger.warning("Using Python approximation - results may differ from R TriMatch")
        import pandas as pd
        from sa_aki_pipeline.fluid import run_diuretic_psm_python
        
        df = pd.read_csv(args.input_csv)
        result = run_diuretic_psm_python(df, config)
        result.matched_data.to_csv(args.output_csv, index=False)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Matching Summary")
    logger.info("=" * 60)
    
    for phenotype, sizes in result.sample_sizes.items():
        if phenotype == 'overall':
            logger.info(f"\nOverall:")
            logger.info(f"  Before matching: {sizes['before']}")
            logger.info(f"  After matching: {sizes['after']}")
            logger.info(f"  Retention: {sizes['after']/sizes['before']*100:.1f}%")
        else:
            logger.info(f"\n{phenotype}:")
            logger.info(f"  Before: {sizes['before']}")
            logger.info(f"  After: {sizes['after']}")
    
    logger.info("\n✓ Matching complete!")
    logger.info(f"Results saved to: {args.output_csv}")


if __name__ == '__main__':
    main()
