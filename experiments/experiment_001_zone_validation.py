"""
Experiment 001: ACWR Zone Boundary Validation
==============================================

Objective:
    Validate the Delta V equation's ACWR zone boundaries using
    real-world injury data from the Mid-Long Distance Runners dataset.

Hypothesis:
    H1: High ACWR (>= 1.5) is associated with increased injury risk
    H2: The optimal zone (0.8-1.3) has the lowest injury rate
    H3: Zone boundaries [0.8, 1.3, 1.5, 2.0] effectively stratify risk

Methods:
    1. Load Mid-Long Distance Runners dataset (74 athletes, 42,766 days)
    2. Calculate ACWR using 7-day acute / 28-day chronic rolling means
    3. Stratify data by ACWR zones
    4. Calculate injury rates and relative risk with 95% CI
    5. Perform train/test split to validate generalization
    6. Test alternative zone boundaries

Statistical Analysis:
    - Fisher's exact test for relative risk significance
    - Confidence intervals using log transformation
    - Multiple threshold sensitivity analysis

Author: Claude (AI Research Assistant)
Date: 2026-01-02
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import json
from typing import Dict, List, Tuple, Any

from data.runners_loader import RunnersDataLoader
from core.delta_v import DeltaVParams, classify_acwr_zone


class Experiment001:
    """
    Experiment 001: ACWR Zone Boundary Validation

    Tests whether the Delta V zone boundaries effectively stratify
    injury risk in real-world data.
    """

    def __init__(self, output_dir: str = None):
        """Initialize experiment."""
        if output_dir is None:
            output_dir = Path(__file__).parent / "results"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.loader = None
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def run(self, verbose: bool = True) -> Dict:
        """
        Run the complete experiment.

        Returns:
            Dictionary with all results
        """
        if verbose:
            print("=" * 70)
            print("EXPERIMENT 001: ACWR ZONE BOUNDARY VALIDATION")
            print("=" * 70)
            print(f"Timestamp: {self.timestamp}")
            print()

        # Step 1: Load data
        self._step_1_load_data(verbose)

        # Step 2: Full dataset analysis
        self._step_2_full_dataset_analysis(verbose)

        # Step 3: Train/test split validation
        self._step_3_train_test_validation(verbose)

        # Step 4: Alternative boundary testing
        self._step_4_alternative_boundaries(verbose)

        # Step 5: Sensitivity analysis
        self._step_5_sensitivity_analysis(verbose)

        # Step 6: Save results
        self._step_6_save_results(verbose)

        return self.results

    def _step_1_load_data(self, verbose: bool = True):
        """Step 1: Load and prepare data."""
        if verbose:
            print("-" * 70)
            print("STEP 1: LOADING DATA")
            print("-" * 70)

        self.loader = RunnersDataLoader()
        self.data = self.loader.load(verbose=verbose)

        self.results['dataset'] = {
            'name': 'Mid-Long Distance Runners',
            'source': 'Kaggle',
            'n_athletes': self.loader.stats.n_athletes,
            'n_days': self.loader.stats.n_days,
            'n_injuries': self.loader.stats.n_injuries,
            'injury_rate_pct': self.loader.stats.injury_rate,
            'avg_days_per_athlete': self.loader.stats.avg_days_per_athlete
        }

    def _step_2_full_dataset_analysis(self, verbose: bool = True):
        """Step 2: Analyze injury rates by zone on full dataset."""
        if verbose:
            print("\n" + "-" * 70)
            print("STEP 2: FULL DATASET ZONE ANALYSIS")
            print("-" * 70)

        # Default Delta V zone boundaries
        default_params = DeltaVParams()
        boundaries = [
            default_params.threshold_low,
            default_params.threshold_optimal_high,
            default_params.threshold_caution,
            default_params.threshold_critical
        ]

        # Get zone analysis
        zone_analysis = self.loader.get_acwr_injury_analysis(boundaries)

        if verbose:
            print("\nInjury Rates by ACWR Zone:")
            print(zone_analysis.to_string(index=False))

        self.results['zone_analysis'] = zone_analysis.to_dict('records')
        self.results['zone_boundaries'] = boundaries

        # Calculate relative risk for high ACWR
        for threshold in [1.3, 1.5, 2.0]:
            rr = self.loader.calculate_relative_risk_ci(threshold=threshold)
            self.results[f'relative_risk_{threshold}'] = rr

            if verbose:
                print(f"\nRelative Risk (ACWR >= {threshold}):")
                print(f"  RR = {rr['relative_risk']:.3f}")
                print(f"  95% CI: [{rr['ci_lower']:.3f}, {rr['ci_upper']:.3f}]")
                print(f"  p-value: {rr['p_value']:.2e}")
                sig = "YES" if rr['statistically_significant'] else "NO"
                print(f"  Statistically significant: {sig}")

    def _step_3_train_test_validation(self, verbose: bool = True):
        """Step 3: Validate on held-out test set."""
        if verbose:
            print("\n" + "-" * 70)
            print("STEP 3: TRAIN/TEST SPLIT VALIDATION")
            print("-" * 70)

        # 70/30 athlete split
        train_ids, test_ids = self.loader.train_test_split_athletes(
            test_size=0.3, random_state=42
        )

        train_df, test_df = self.loader.get_train_test_data(train_ids, test_ids)

        if verbose:
            print(f"\nTrain set: {len(train_ids)} athletes, {len(train_df):,} days")
            print(f"Test set: {len(test_ids)} athletes, {len(test_df):,} days")

        # Calculate RR on train and test separately
        train_test_results = {}

        for name, df in [('train', train_df), ('test', test_df)]:
            df_clean = df.dropna(subset=['acwr'])

            # Split by threshold
            high_acwr = df_clean[df_clean['acwr'] >= 1.5]
            low_acwr = df_clean[df_clean['acwr'] < 1.5]

            a = high_acwr['injury'].sum()
            b = len(high_acwr) - a
            c = low_acwr['injury'].sum()
            d = len(low_acwr) - c

            rate_high = a / (a + b) if (a + b) > 0 else 0
            rate_low = c / (c + d) if (c + d) > 0 else 0
            rr = rate_high / rate_low if rate_low > 0 else np.inf

            # CI calculation
            if a > 0 and c > 0:
                se_log_rr = np.sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
                z = 1.96
                log_rr = np.log(rr)
                ci_lower = np.exp(log_rr - z * se_log_rr)
                ci_upper = np.exp(log_rr + z * se_log_rr)
            else:
                ci_lower = np.nan
                ci_upper = np.nan

            # P-value
            table = [[a, b], [c, d]]
            _, p_value = stats.fisher_exact(table)

            train_test_results[name] = {
                'n_athletes': len(train_ids) if name == 'train' else len(test_ids),
                'n_days': len(df_clean),
                'n_injuries': int(a + c),
                'relative_risk': rr,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'p_value': p_value,
                'significant': p_value < 0.05
            }

            if verbose:
                print(f"\n{name.upper()} SET:")
                print(f"  RR = {rr:.3f} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])")
                print(f"  p-value: {p_value:.2e}")
                print(f"  Significant: {'YES' if p_value < 0.05 else 'NO'}")

        self.results['train_test_validation'] = train_test_results

        # Check for overfitting
        train_rr = train_test_results['train']['relative_risk']
        test_rr = train_test_results['test']['relative_risk']

        if not np.isinf(train_rr) and not np.isinf(test_rr):
            rr_diff = abs(train_rr - test_rr) / train_rr * 100
            self.results['overfitting_check'] = {
                'train_rr': train_rr,
                'test_rr': test_rr,
                'relative_difference_pct': rr_diff,
                'overfitting_concern': rr_diff > 50  # >50% difference = concern
            }

            if verbose:
                print(f"\nOVERFITTING CHECK:")
                print(f"  Train RR: {train_rr:.3f}")
                print(f"  Test RR: {test_rr:.3f}")
                print(f"  Difference: {rr_diff:.1f}%")
                concern = "YES" if rr_diff > 50 else "NO"
                print(f"  Overfitting concern: {concern}")

    def _step_4_alternative_boundaries(self, verbose: bool = True):
        """Step 4: Test alternative zone boundaries."""
        if verbose:
            print("\n" + "-" * 70)
            print("STEP 4: ALTERNATIVE ZONE BOUNDARIES")
            print("-" * 70)

        # Test different boundary configurations
        boundary_configs = {
            'default': [0.8, 1.3, 1.5, 2.0],
            'conservative': [0.7, 1.2, 1.4, 1.8],
            'aggressive': [0.85, 1.35, 1.6, 2.2],
            'narrow_optimal': [0.85, 1.2, 1.5, 2.0],
            'wide_optimal': [0.7, 1.4, 1.5, 2.0],
        }

        boundary_results = {}

        for name, boundaries in boundary_configs.items():
            analysis = self.loader.get_acwr_injury_analysis(boundaries)

            # Get RR for high zone
            threshold = boundaries[2]  # caution threshold
            rr = self.loader.calculate_relative_risk_ci(threshold=threshold)

            boundary_results[name] = {
                'boundaries': boundaries,
                'relative_risk': rr['relative_risk'],
                'ci_lower': rr['ci_lower'],
                'ci_upper': rr['ci_upper'],
                'p_value': rr['p_value'],
                'significant': rr['statistically_significant']
            }

            if verbose:
                print(f"\n{name.upper()} boundaries: {boundaries}")
                print(f"  RR (>= {threshold}): {rr['relative_risk']:.3f}")
                print(f"  p-value: {rr['p_value']:.2e}")

        self.results['alternative_boundaries'] = boundary_results

    def _step_5_sensitivity_analysis(self, verbose: bool = True):
        """Step 5: Sensitivity analysis across threshold range."""
        if verbose:
            print("\n" + "-" * 70)
            print("STEP 5: THRESHOLD SENSITIVITY ANALYSIS")
            print("-" * 70)

        thresholds = np.arange(1.0, 2.5, 0.1)
        sensitivity = []

        for threshold in thresholds:
            rr = self.loader.calculate_relative_risk_ci(threshold=threshold)
            sensitivity.append({
                'threshold': threshold,
                'relative_risk': rr['relative_risk'],
                'ci_lower': rr['ci_lower'],
                'ci_upper': rr['ci_upper'],
                'p_value': rr['p_value'],
                'n_high': rr['n_high_acwr'],
                'injuries_high': rr['injuries_high_acwr']
            })

        self.results['sensitivity_analysis'] = sensitivity

        if verbose:
            print("\nThreshold -> Relative Risk:")
            for s in sensitivity:
                sig = "*" if s['p_value'] < 0.05 else ""
                print(f"  ACWR >= {s['threshold']:.1f}: RR = {s['relative_risk']:.2f} "
                      f"(n={s['n_high']:,}, injuries={s['injuries_high']}){sig}")

        # Find optimal threshold (highest significant RR)
        significant = [s for s in sensitivity if s['p_value'] < 0.05]
        if significant:
            optimal = max(significant, key=lambda x: x['relative_risk'])
            self.results['optimal_threshold'] = optimal['threshold']
            if verbose:
                print(f"\nOptimal threshold (highest significant RR): {optimal['threshold']:.1f}")
                print(f"  RR = {optimal['relative_risk']:.3f}")

    def _step_6_save_results(self, verbose: bool = True):
        """Step 6: Save results to files."""
        if verbose:
            print("\n" + "-" * 70)
            print("STEP 6: SAVING RESULTS")
            print("-" * 70)

        # Convert numpy types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                if np.isnan(obj):
                    return None
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(i) for i in obj]
            return obj

        results_clean = convert_types(self.results)

        # Save JSON results
        output_file = self.output_dir / f"experiment_001_{self.timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(results_clean, f, indent=2)

        if verbose:
            print(f"Results saved to: {output_file}")

        # Generate summary report
        self._generate_report(verbose)

        return output_file

    def _generate_report(self, verbose: bool = True):
        """Generate human-readable report."""
        report = f"""
================================================================================
EXPERIMENT 001: ACWR ZONE BOUNDARY VALIDATION
================================================================================

Date: {self.timestamp}

DATASET
-------
Name: {self.results['dataset']['name']}
Athletes: {self.results['dataset']['n_athletes']}
Days: {self.results['dataset']['n_days']:,}
Injuries: {self.results['dataset']['n_injuries']}
Injury Rate: {self.results['dataset']['injury_rate_pct']:.3f}%

ZONE BOUNDARIES TESTED
----------------------
Default: [0.8, 1.3, 1.5, 2.0]

PRIMARY RESULTS
---------------
"""
        # Add RR for key thresholds
        for thresh in [1.3, 1.5, 2.0]:
            key = f'relative_risk_{thresh}'
            if key in self.results:
                rr = self.results[key]
                sig = "YES" if rr['statistically_significant'] else "NO"
                report += f"""
ACWR >= {thresh}:
  Relative Risk: {rr['relative_risk']:.3f}
  95% CI: [{rr['ci_lower']:.3f}, {rr['ci_upper']:.3f}]
  p-value: {rr['p_value']:.2e}
  Significant: {sig}
  N (high ACWR): {rr['n_high_acwr']:,}
  Injuries (high ACWR): {rr['injuries_high_acwr']}
"""

        # Train/test validation
        if 'train_test_validation' in self.results:
            tt = self.results['train_test_validation']
            report += f"""
TRAIN/TEST VALIDATION
--------------------
Train: RR = {tt['train']['relative_risk']:.3f}, p = {tt['train']['p_value']:.2e}
Test:  RR = {tt['test']['relative_risk']:.3f}, p = {tt['test']['p_value']:.2e}
"""

        # Overfitting check
        if 'overfitting_check' in self.results:
            oc = self.results['overfitting_check']
            concern = "YES" if oc['overfitting_concern'] else "NO"
            report += f"""
OVERFITTING CHECK
-----------------
Train RR: {oc['train_rr']:.3f}
Test RR: {oc['test_rr']:.3f}
Difference: {oc['relative_difference_pct']:.1f}%
Concern: {concern}
"""

        # Conclusions
        report += """
CONCLUSIONS
-----------
"""
        # Add conclusions based on results
        rr_1_5 = self.results.get('relative_risk_1.5', {})
        if rr_1_5.get('statistically_significant'):
            report += f"1. HIGH ACWR (>= 1.5) IS ASSOCIATED WITH INCREASED INJURY RISK\n"
            report += f"   - Relative Risk: {rr_1_5['relative_risk']:.2f}x\n"
            report += f"   - This finding is statistically significant (p < 0.05)\n"
        else:
            report += "1. High ACWR association with injury NOT statistically significant\n"

        if 'train_test_validation' in self.results:
            if self.results['train_test_validation']['test']['significant']:
                report += "2. FINDING VALIDATED ON HELD-OUT TEST SET\n"
                report += "   - Effect generalizes to unseen athletes\n"
            else:
                report += "2. Finding NOT validated on test set\n"

        if 'overfitting_check' in self.results:
            if not self.results['overfitting_check']['overfitting_concern']:
                report += "3. NO OVERFITTING CONCERN\n"
                report += "   - Train and test RR are similar\n"
            else:
                report += "3. POTENTIAL OVERFITTING\n"
                report += "   - Large difference between train and test RR\n"

        report += """
================================================================================
"""

        # Save report
        report_file = self.output_dir / f"experiment_001_{self.timestamp}_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)

        if verbose:
            print(report)
            print(f"\nReport saved to: {report_file}")


def main():
    """Run the experiment."""
    exp = Experiment001()
    results = exp.run(verbose=True)
    return results


if __name__ == "__main__":
    main()
