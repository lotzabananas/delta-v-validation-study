"""
Experiment 002: Lagged ACWR-Injury Analysis
============================================

Objective:
    Investigate the temporal relationship between ACWR and injury.
    Experiment 001 showed HIGH ACWR associated with LOWER injury rates,
    which contradicts established literature. This suggests:

    1. Reverse causality: Injury causes LOW ACWR (athletes reduce training)
    2. Need to look at ACWR BEFORE injury, not ON injury day

Methods:
    1. For each injury event, extract ACWR in days BEFORE the injury
    2. Compare pre-injury ACWR distribution to baseline
    3. Calculate relative risk using lagged ACWR values
    4. Identify optimal lag window for prediction

Statistical Analysis:
    - Compare ACWR distributions (injured vs non-injured periods)
    - Cox proportional hazards for time-to-injury analysis
    - AUC-ROC for ACWR as injury predictor at different lags

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
from core.delta_v import DeltaVParams


class Experiment002:
    """
    Experiment 002: Lagged ACWR-Injury Analysis

    Examines ACWR in the days BEFORE injuries occur.
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
        """Run the complete experiment."""
        if verbose:
            print("=" * 70)
            print("EXPERIMENT 002: LAGGED ACWR-INJURY ANALYSIS")
            print("=" * 70)
            print(f"Timestamp: {self.timestamp}")
            print()

        # Step 1: Load data
        self._step_1_load_data(verbose)

        # Step 2: Analyze injury patterns
        self._step_2_injury_patterns(verbose)

        # Step 3: Lagged ACWR analysis
        self._step_3_lagged_acwr(verbose)

        # Step 4: Pre-injury vs baseline comparison
        self._step_4_pre_injury_comparison(verbose)

        # Step 5: Optimal lag window
        self._step_5_optimal_lag(verbose)

        # Step 6: Revised zone validation
        self._step_6_revised_validation(verbose)

        # Step 7: Save results
        self._step_7_save_results(verbose)

        return self.results

    def _step_1_load_data(self, verbose: bool = True):
        """Load data."""
        if verbose:
            print("-" * 70)
            print("STEP 1: LOADING DATA")
            print("-" * 70)

        self.loader = RunnersDataLoader()
        self.data = self.loader.load(verbose=verbose)

        self.results['dataset'] = {
            'name': 'Mid-Long Distance Runners',
            'n_athletes': self.loader.stats.n_athletes,
            'n_days': self.loader.stats.n_days,
            'n_injuries': self.loader.stats.n_injuries,
        }

    def _step_2_injury_patterns(self, verbose: bool = True):
        """Analyze injury occurrence patterns."""
        if verbose:
            print("\n" + "-" * 70)
            print("STEP 2: INJURY PATTERNS")
            print("-" * 70)

        df = self.data

        # Count consecutive injury days (injury events vs injury days)
        injury_events = []

        for athlete_id in df['athlete_id'].unique():
            athlete_df = df[df['athlete_id'] == athlete_id].sort_values('date_index')

            in_injury = False
            event_start = None

            for _, row in athlete_df.iterrows():
                if row['injury'] == 1 and not in_injury:
                    # Start of new injury event
                    in_injury = True
                    event_start = row['date_index']
                elif row['injury'] == 0 and in_injury:
                    # End of injury event
                    in_injury = False
                    injury_events.append({
                        'athlete_id': athlete_id,
                        'start_day': event_start,
                        'end_day': row['date_index'] - 1,
                        'duration': row['date_index'] - event_start
                    })

        injury_events_df = pd.DataFrame(injury_events)

        if len(injury_events_df) > 0:
            self.results['injury_events'] = {
                'n_events': len(injury_events_df),
                'mean_duration_days': injury_events_df['duration'].mean(),
                'median_duration_days': injury_events_df['duration'].median(),
                'max_duration_days': injury_events_df['duration'].max(),
            }

            if verbose:
                print(f"\nInjury Event Analysis:")
                print(f"  Total injury events: {len(injury_events_df)}")
                print(f"  Total injury DAYS: {df['injury'].sum()}")
                print(f"  Mean duration: {injury_events_df['duration'].mean():.1f} days")
                print(f"  Median duration: {injury_events_df['duration'].median():.0f} days")

                # Show distribution
                print(f"\n  Duration distribution:")
                for dur in [1, 2, 3, 5, 7, 14, 30]:
                    pct = (injury_events_df['duration'] <= dur).mean() * 100
                    print(f"    <= {dur} days: {pct:.1f}%")

        self.injury_events_df = injury_events_df

    def _step_3_lagged_acwr(self, verbose: bool = True):
        """Analyze ACWR at different lags before injury."""
        if verbose:
            print("\n" + "-" * 70)
            print("STEP 3: LAGGED ACWR ANALYSIS")
            print("-" * 70)

        df = self.data

        # For each injury START, get ACWR at different lags
        lagged_results = {}

        for lag in range(0, 15):  # 0 to 14 days before injury
            pre_injury_acwr = []
            non_injury_acwr = []

            for athlete_id in df['athlete_id'].unique():
                athlete_df = df[df['athlete_id'] == athlete_id].sort_values('date_index')

                # Find injury starts (first day of injury = 1 after day of injury = 0)
                athlete_df = athlete_df.reset_index(drop=True)

                for i, row in athlete_df.iterrows():
                    if i == 0:
                        continue

                    # Check if this is the START of an injury
                    if row['injury'] == 1 and athlete_df.iloc[i-1]['injury'] == 0:
                        # Get ACWR at 'lag' days before
                        target_idx = i - lag
                        if target_idx >= 0:
                            acwr_val = athlete_df.iloc[target_idx]['acwr']
                            if pd.notna(acwr_val):
                                pre_injury_acwr.append(acwr_val)
                    elif row['injury'] == 0:
                        # Non-injury day
                        acwr_val = row['acwr']
                        if pd.notna(acwr_val):
                            non_injury_acwr.append(acwr_val)

            if len(pre_injury_acwr) > 0 and len(non_injury_acwr) > 0:
                # Compare distributions
                stat, p_value = stats.mannwhitneyu(
                    pre_injury_acwr, non_injury_acwr, alternative='two-sided'
                )

                # Calculate means
                pre_mean = np.mean(pre_injury_acwr)
                non_mean = np.mean(non_injury_acwr)

                lagged_results[lag] = {
                    'lag_days': lag,
                    'n_pre_injury': len(pre_injury_acwr),
                    'n_non_injury': len(non_injury_acwr),
                    'pre_injury_mean_acwr': pre_mean,
                    'non_injury_mean_acwr': non_mean,
                    'acwr_difference': pre_mean - non_mean,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }

        self.results['lagged_analysis'] = lagged_results

        if verbose:
            print("\nACWR at different lags before injury:")
            print(f"{'Lag':>4} {'Pre-Inj ACWR':>12} {'Non-Inj ACWR':>13} {'Diff':>8} {'p-value':>12} {'Sig':>5}")
            print("-" * 60)
            for lag, res in sorted(lagged_results.items()):
                sig = "*" if res['significant'] else ""
                print(f"{lag:>4}d {res['pre_injury_mean_acwr']:>12.3f} {res['non_injury_mean_acwr']:>13.3f} "
                      f"{res['acwr_difference']:>+8.3f} {res['p_value']:>12.2e} {sig:>5}")

    def _step_4_pre_injury_comparison(self, verbose: bool = True):
        """Compare ACWR in week before injury to baseline."""
        if verbose:
            print("\n" + "-" * 70)
            print("STEP 4: PRE-INJURY WEEK COMPARISON")
            print("-" * 70)

        df = self.data

        # Extract week-before-injury ACWR values
        pre_injury_week_acwr = []
        control_acwr = []

        for athlete_id in df['athlete_id'].unique():
            athlete_df = df[df['athlete_id'] == athlete_id].sort_values('date_index')
            athlete_df = athlete_df.reset_index(drop=True)

            injury_start_indices = []

            for i, row in athlete_df.iterrows():
                if i == 0:
                    continue
                if row['injury'] == 1 and athlete_df.iloc[i-1]['injury'] == 0:
                    injury_start_indices.append(i)

            # Get ACWR from days 1-7 before each injury
            for inj_idx in injury_start_indices:
                for offset in range(1, 8):  # Days 1-7 before injury
                    target_idx = inj_idx - offset
                    if target_idx >= 0:
                        acwr_val = athlete_df.iloc[target_idx]['acwr']
                        if pd.notna(acwr_val):
                            pre_injury_week_acwr.append(acwr_val)

            # Control: Non-injury days not within 7 days of injury
            injury_adjacent = set()
            for inj_idx in injury_start_indices:
                for offset in range(-7, 8):
                    injury_adjacent.add(inj_idx + offset)

            for i, row in athlete_df.iterrows():
                if row['injury'] == 0 and i not in injury_adjacent:
                    if pd.notna(row['acwr']):
                        control_acwr.append(row['acwr'])

        # Statistical comparison
        if len(pre_injury_week_acwr) > 10 and len(control_acwr) > 10:
            stat, p_value = stats.mannwhitneyu(
                pre_injury_week_acwr, control_acwr, alternative='two-sided'
            )

            pre_mean = np.mean(pre_injury_week_acwr)
            control_mean = np.mean(control_acwr)

            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(
                (np.var(pre_injury_week_acwr) + np.var(control_acwr)) / 2
            )
            cohens_d = (pre_mean - control_mean) / pooled_std if pooled_std > 0 else 0

            self.results['pre_injury_comparison'] = {
                'n_pre_injury_week': len(pre_injury_week_acwr),
                'n_control': len(control_acwr),
                'pre_injury_mean_acwr': pre_mean,
                'control_mean_acwr': control_mean,
                'difference': pre_mean - control_mean,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significant': p_value < 0.05
            }

            if verbose:
                print(f"\nWeek before injury vs baseline:")
                print(f"  Pre-injury week ACWR: {pre_mean:.3f} (n={len(pre_injury_week_acwr):,})")
                print(f"  Control ACWR: {control_mean:.3f} (n={len(control_acwr):,})")
                print(f"  Difference: {pre_mean - control_mean:+.3f}")
                print(f"  p-value: {p_value:.2e}")
                print(f"  Cohen's d: {cohens_d:.3f}")
                print(f"  Significant: {'YES' if p_value < 0.05 else 'NO'}")

    def _step_5_optimal_lag(self, verbose: bool = True):
        """Find optimal lag for injury prediction."""
        if verbose:
            print("\n" + "-" * 70)
            print("STEP 5: OPTIMAL LAG FOR PREDICTION")
            print("-" * 70)

        df = self.data

        # Calculate AUC at different lags
        from sklearn.metrics import roc_auc_score

        auc_results = {}

        for lag in range(1, 15):
            # Create lagged features
            y_true = []
            y_pred = []

            for athlete_id in df['athlete_id'].unique():
                athlete_df = df[df['athlete_id'] == athlete_id].sort_values('date_index')
                athlete_df = athlete_df.reset_index(drop=True)

                for i, row in athlete_df.iterrows():
                    if i < lag:
                        continue

                    # Use ACWR from 'lag' days ago to predict today's injury status
                    acwr_lagged = athlete_df.iloc[i - lag]['acwr']
                    injury_today = row['injury']

                    if pd.notna(acwr_lagged):
                        y_pred.append(acwr_lagged)
                        y_true.append(injury_today)

            if len(y_true) > 100 and sum(y_true) > 10:
                try:
                    auc = roc_auc_score(y_true, y_pred)
                    auc_results[lag] = {
                        'lag': lag,
                        'auc': auc,
                        'n_samples': len(y_true),
                        'n_injuries': sum(y_true)
                    }
                except:
                    pass

        if auc_results:
            best_lag = max(auc_results.keys(), key=lambda x: auc_results[x]['auc'])
            self.results['optimal_lag'] = auc_results[best_lag]
            self.results['all_lag_aucs'] = auc_results

            if verbose:
                print("\nAUC-ROC by lag (ACWR predicting injury):")
                print(f"{'Lag':>4} {'AUC':>8} {'N':>10} {'Injuries':>10}")
                print("-" * 40)
                for lag, res in sorted(auc_results.items()):
                    marker = " <-- BEST" if lag == best_lag else ""
                    print(f"{lag:>4}d {res['auc']:>8.4f} {res['n_samples']:>10,} {res['n_injuries']:>10}{marker}")

                print(f"\nNote: AUC > 0.5 means HIGH ACWR predicts injury")
                print(f"      AUC < 0.5 means LOW ACWR predicts injury")

    def _step_6_revised_validation(self, verbose: bool = True):
        """Revised zone validation using lagged ACWR."""
        if verbose:
            print("\n" + "-" * 70)
            print("STEP 6: REVISED ZONE VALIDATION (LAGGED)")
            print("-" * 70)

        df = self.data

        # Use 7-day lagged ACWR to predict injuries
        lag = 7  # Use week-ahead prediction

        high_acwr_injuries = 0
        high_acwr_total = 0
        low_acwr_injuries = 0
        low_acwr_total = 0

        for athlete_id in df['athlete_id'].unique():
            athlete_df = df[df['athlete_id'] == athlete_id].sort_values('date_index')
            athlete_df = athlete_df.reset_index(drop=True)

            for i, row in athlete_df.iterrows():
                if i < lag:
                    continue

                acwr_lagged = athlete_df.iloc[i - lag]['acwr']
                injury_today = row['injury']

                if pd.notna(acwr_lagged):
                    if acwr_lagged >= 1.5:
                        high_acwr_total += 1
                        high_acwr_injuries += injury_today
                    else:
                        low_acwr_total += 1
                        low_acwr_injuries += injury_today

        # Calculate relative risk
        rate_high = high_acwr_injuries / high_acwr_total if high_acwr_total > 0 else 0
        rate_low = low_acwr_injuries / low_acwr_total if low_acwr_total > 0 else 0
        rr = rate_high / rate_low if rate_low > 0 else np.inf

        # CI and p-value
        a, b = high_acwr_injuries, high_acwr_total - high_acwr_injuries
        c, d = low_acwr_injuries, low_acwr_total - low_acwr_injuries

        if a > 0 and c > 0:
            se_log_rr = np.sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
            z = 1.96
            log_rr = np.log(rr)
            ci_lower = np.exp(log_rr - z * se_log_rr)
            ci_upper = np.exp(log_rr + z * se_log_rr)
        else:
            ci_lower = ci_upper = np.nan

        _, p_value = stats.fisher_exact([[a, b], [c, d]])

        self.results['lagged_relative_risk'] = {
            'lag_days': lag,
            'threshold': 1.5,
            'n_high_acwr': high_acwr_total,
            'n_low_acwr': low_acwr_total,
            'injuries_high_acwr': high_acwr_injuries,
            'injuries_low_acwr': low_acwr_injuries,
            'rate_high': rate_high,
            'rate_low': rate_low,
            'relative_risk': rr,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value,
            'significant': p_value < 0.05
        }

        if verbose:
            print(f"\nRelative Risk (ACWR >= 1.5, {lag}-day lag):")
            print(f"  High ACWR (>= 1.5): {high_acwr_injuries}/{high_acwr_total} injuries ({rate_high*100:.3f}%)")
            print(f"  Low ACWR (< 1.5): {low_acwr_injuries}/{low_acwr_total} injuries ({rate_low*100:.3f}%)")
            print(f"  Relative Risk: {rr:.3f}")
            print(f"  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
            print(f"  p-value: {p_value:.2e}")
            print(f"  Significant: {'YES' if p_value < 0.05 else 'NO'}")

    def _step_7_save_results(self, verbose: bool = True):
        """Save results."""
        if verbose:
            print("\n" + "-" * 70)
            print("STEP 7: SAVING RESULTS")
            print("-" * 70)

        # Convert types for JSON
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
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(i) for i in obj]
            return obj

        results_clean = convert_types(self.results)

        output_file = self.output_dir / f"experiment_002_{self.timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(results_clean, f, indent=2)

        if verbose:
            print(f"Results saved to: {output_file}")

        # Generate report
        self._generate_report(verbose)

    def _generate_report(self, verbose: bool = True):
        """Generate report."""
        report = f"""
================================================================================
EXPERIMENT 002: LAGGED ACWR-INJURY ANALYSIS
================================================================================

Date: {self.timestamp}

OBJECTIVE
---------
Experiment 001 showed counterintuitive results: HIGH ACWR was associated
with LOWER injury rates. This experiment investigates whether this is due
to reverse causality (injury -> reduced training -> low ACWR).

KEY FINDINGS
------------
"""
        # Pre-injury comparison
        if 'pre_injury_comparison' in self.results:
            comp = self.results['pre_injury_comparison']
            report += f"""
1. PRE-INJURY WEEK ACWR VS BASELINE:
   - Pre-injury week ACWR: {comp['pre_injury_mean_acwr']:.3f}
   - Control (non-injury) ACWR: {comp['control_mean_acwr']:.3f}
   - Difference: {comp['difference']:+.3f}
   - p-value: {comp['p_value']:.2e}
   - Cohen's d: {comp['cohens_d']:.3f}
   - Significant: {'YES' if comp['significant'] else 'NO'}
"""
            if comp['difference'] > 0:
                report += "   => ACWR IS HIGHER in the week before injury!\n"
            else:
                report += "   => ACWR IS LOWER in the week before injury.\n"

        # Lagged RR
        if 'lagged_relative_risk' in self.results:
            rr = self.results['lagged_relative_risk']
            report += f"""
2. LAGGED RELATIVE RISK (7-day lag):
   - Relative Risk: {rr['relative_risk']:.3f}
   - 95% CI: [{rr['ci_lower']:.3f}, {rr['ci_upper']:.3f}]
   - p-value: {rr['p_value']:.2e}
"""
            if rr['relative_risk'] > 1:
                report += "   => HIGH ACWR (7 days prior) INCREASES injury risk!\n"
            else:
                report += "   => HIGH ACWR (7 days prior) DECREASES injury risk.\n"

        # Optimal lag
        if 'optimal_lag' in self.results:
            opt = self.results['optimal_lag']
            report += f"""
3. OPTIMAL PREDICTION LAG:
   - Best lag: {opt['lag']} days
   - AUC: {opt['auc']:.4f}
"""
            if opt['auc'] > 0.5:
                report += "   => HIGHER ACWR predicts MORE injuries\n"
            else:
                report += "   => HIGHER ACWR predicts FEWER injuries\n"

        report += """
CONCLUSIONS
-----------
"""
        # Add interpretation
        if 'pre_injury_comparison' in self.results and 'lagged_relative_risk' in self.results:
            comp = self.results['pre_injury_comparison']
            rr = self.results['lagged_relative_risk']

            if comp['difference'] > 0 and rr['relative_risk'] > 1:
                report += """
The data SUPPORTS the ACWR-injury hypothesis when using proper temporal analysis:
- ACWR IS higher in the week before injuries
- High ACWR (7 days prior) IS associated with increased injury risk

The initial counterintuitive results were due to reverse causality:
injured athletes have LOW ACWR because they reduce training after injury.
"""
            else:
                report += """
Even with lagged analysis, the ACWR-injury relationship is not as expected.
This suggests:
- The ACWR paradigm may not apply to this population
- There may be other confounding factors
- The injury labels may not capture the type of injuries ACWR predicts
"""

        report += """
================================================================================
"""

        report_file = self.output_dir / f"experiment_002_{self.timestamp}_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)

        if verbose:
            print(report)


def main():
    """Run the experiment."""
    exp = Experiment002()
    results = exp.run(verbose=True)
    return results


if __name__ == "__main__":
    main()
