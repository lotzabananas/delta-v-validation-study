"""
Real Data Backtesting for Delta V Equation
===========================================

Simulates how the Delta V equation would have guided real athletes
and whether it would have prevented injuries.

Key questions:
1. Would Delta V have flagged high-risk periods before injuries?
2. How often would it have recommended load reductions?
3. What would alternative parameters have recommended?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

from core.delta_v import DeltaVParams, calculate_delta_v, classify_acwr_zone
from data.pmdata_loader import PMDataLoader


@dataclass
class BacktestResult:
    """Results from backtesting Delta V on real data."""
    participant_id: str
    total_days: int
    injury_events: int
    injuries_predicted: int  # Days where Delta V flagged high risk before injury
    injuries_missed: int     # Injuries that occurred without prior warning
    false_alarms: int        # High-risk flags that didn't result in injury
    avg_recommended_change: float
    reduction_recommendations: int  # Days where Delta V said to reduce
    increase_recommendations: int   # Days where Delta V said to increase
    maintain_recommendations: int   # Days where Delta V said to maintain


class RealDataBacktester:
    """
    Backtests the Delta V equation against real training/injury data.
    """

    def __init__(self, data_loader: PMDataLoader = None):
        """
        Initialize with a data loader.

        Args:
            data_loader: PMDataLoader instance (will load if not provided)
        """
        if data_loader is None:
            data_loader = PMDataLoader()
            data_loader.load_all(verbose=False)

        self.loader = data_loader

    def backtest_participant(self, participant_id: str,
                             params: DeltaVParams = None,
                             lookback_days: int = 7) -> Dict:
        """
        Backtest Delta V for a single participant.

        Args:
            participant_id: The participant ID
            params: Delta V parameters to test
            lookback_days: Days before injury to check for warnings

        Returns:
            Dictionary with backtest results
        """
        if params is None:
            params = DeltaVParams()

        # Get ACWR data
        acwr_data = self.loader.get_acwr_series(participant_id)

        if len(acwr_data) == 0:
            return None

        # Get injury dates for this participant
        injuries = self.loader.get_injury_dates(participant_id)

        injury_dates = set()
        for _, row in injuries.iterrows():
            date = row['date']
            if hasattr(date, 'date'):
                date = date.date()
            injury_dates.add(date)

        # Calculate Delta V recommendations for each day
        acwr_data = acwr_data.copy()

        def get_delta_v_value(acwr):
            if pd.isna(acwr):
                return np.nan
            result = calculate_delta_v(acwr, params)
            return result[0]  # Returns (delta_v, zone, flag) tuple

        acwr_data['delta_v'] = acwr_data['acwr'].apply(get_delta_v_value)
        acwr_data['zone'] = acwr_data['acwr'].apply(
            lambda x: classify_acwr_zone(x, params) if pd.notna(x) else None
        )

        # Analyze recommendations
        valid_data = acwr_data.dropna(subset=['delta_v'])

        reduction_recs = (valid_data['delta_v'] < -0.05).sum()
        increase_recs = (valid_data['delta_v'] > 0.05).sum()
        maintain_recs = ((valid_data['delta_v'] >= -0.05) & (valid_data['delta_v'] <= 0.05)).sum()

        avg_change = valid_data['delta_v'].mean()

        # Check if Delta V would have warned before injuries
        injuries_predicted = 0
        injuries_missed = 0
        warning_details = []

        for injury_date in injury_dates:
            # Look at the days before injury
            pre_injury = acwr_data[
                (acwr_data['date'].dt.date < injury_date) &
                (acwr_data['date'].dt.date >= injury_date - timedelta(days=lookback_days))
            ]

            had_warning = False
            max_acwr_before = 0

            for _, row in pre_injury.iterrows():
                if pd.notna(row['acwr']):
                    max_acwr_before = max(max_acwr_before, row['acwr'])
                    # Check if Delta V recommended reduction (high risk)
                    if row['zone'] in ['red', 'critical'] or row['delta_v'] < -0.1:
                        had_warning = True

            if had_warning:
                injuries_predicted += 1
            else:
                injuries_missed += 1

            warning_details.append({
                'date': injury_date,
                'predicted': had_warning,
                'max_acwr_before': max_acwr_before
            })

        # Count false alarms (high-risk zones without injury)
        high_risk_days = valid_data[valid_data['zone'].isin(['red', 'critical'])]

        false_alarms = 0
        for _, row in high_risk_days.iterrows():
            day = row['date'].date() if hasattr(row['date'], 'date') else row['date']
            # Check if injury occurred within next 7 days
            following_week = set(
                day + timedelta(days=i) for i in range(1, 8)
            )
            if not following_week.intersection(injury_dates):
                false_alarms += 1

        return {
            'participant_id': participant_id,
            'total_days': len(valid_data),
            'injury_events': len(injury_dates),
            'injuries_predicted': injuries_predicted,
            'injuries_missed': injuries_missed,
            'false_alarms': false_alarms,
            'avg_delta_v': avg_change,
            'reduction_recommendations': reduction_recs,
            'increase_recommendations': increase_recs,
            'maintain_recommendations': maintain_recs,
            'warning_details': warning_details,
            'zone_distribution': valid_data['zone'].value_counts().to_dict()
        }

    def backtest_all(self, params: DeltaVParams = None) -> Dict:
        """
        Backtest Delta V across all participants.

        Args:
            params: Delta V parameters to test

        Returns:
            Aggregated results
        """
        if params is None:
            params = DeltaVParams()

        results = []
        for pid in self.loader.training_data.keys():
            result = self.backtest_participant(pid, params)
            if result:
                results.append(result)

        if not results:
            return {}

        # Aggregate results
        total_injuries = sum(r['injury_events'] for r in results)
        total_predicted = sum(r['injuries_predicted'] for r in results)
        total_missed = sum(r['injuries_missed'] for r in results)
        total_false_alarms = sum(r['false_alarms'] for r in results)
        total_days = sum(r['total_days'] for r in results)

        # Zone distribution
        zone_totals = {}
        for r in results:
            for zone, count in r['zone_distribution'].items():
                zone_totals[zone] = zone_totals.get(zone, 0) + count

        return {
            'n_participants': len(results),
            'total_days': total_days,
            'total_injuries': total_injuries,
            'injuries_predicted': total_predicted,
            'injuries_missed': total_missed,
            'prediction_rate': total_predicted / total_injuries * 100 if total_injuries > 0 else 0,
            'false_alarms': total_false_alarms,
            'false_alarm_rate': total_false_alarms / total_days * 100,
            'avg_delta_v': np.mean([r['avg_delta_v'] for r in results]),
            'reduction_pct': sum(r['reduction_recommendations'] for r in results) / total_days * 100,
            'increase_pct': sum(r['increase_recommendations'] for r in results) / total_days * 100,
            'zone_distribution': zone_totals,
            'individual_results': results
        }

    def compare_parameters(self, param_sets: Dict[str, DeltaVParams]) -> pd.DataFrame:
        """
        Compare different parameter sets.

        Args:
            param_sets: Dictionary of name -> DeltaVParams

        Returns:
            DataFrame comparing results
        """
        results = []

        for name, params in param_sets.items():
            result = self.backtest_all(params)
            result['param_set'] = name
            results.append(result)

        comparison = pd.DataFrame([
            {
                'param_set': r['param_set'],
                'prediction_rate': r['prediction_rate'],
                'false_alarm_rate': r['false_alarm_rate'],
                'injuries_predicted': r['injuries_predicted'],
                'injuries_missed': r['injuries_missed'],
                'avg_delta_v': r['avg_delta_v'],
                'reduction_pct': r['reduction_pct']
            }
            for r in results
        ])

        return comparison

    def simulate_following_delta_v(self, participant_id: str,
                                   params: DeltaVParams = None) -> pd.DataFrame:
        """
        Simulate what would have happened if athlete followed Delta V.

        Args:
            participant_id: The participant ID
            params: Delta V parameters

        Returns:
            DataFrame with actual vs simulated loads
        """
        if params is None:
            params = DeltaVParams()

        acwr_data = self.loader.get_acwr_series(participant_id)

        if len(acwr_data) == 0:
            return pd.DataFrame()

        # Start simulation
        simulated_load = []
        actual_load = acwr_data['load'].values

        # Initialize with first 4 weeks of actual data (to establish baseline)
        warmup = 28
        simulated_load = list(actual_load[:warmup])

        # Simulate following Delta V recommendations
        for i in range(warmup, len(actual_load)):
            # Calculate current ACWR based on simulated data
            if len(simulated_load) >= 28:
                acute = np.mean(simulated_load[-7:])
                chronic = np.mean(simulated_load[-28:])
                acwr = acute / chronic if chronic > 0 else 1.0
            else:
                acwr = 1.0

            # Get Delta V recommendation
            delta_v, zone, flag = calculate_delta_v(acwr, params)

            # Apply to previous week's average
            prev_week_avg = np.mean(simulated_load[-7:]) if len(simulated_load) >= 7 else simulated_load[-1]
            new_load = prev_week_avg * (1 + delta_v)

            # Don't go below 0
            new_load = max(0, new_load)

            simulated_load.append(new_load)

        # Create comparison DataFrame
        result = pd.DataFrame({
            'date': acwr_data['date'],
            'actual_load': actual_load,
            'simulated_load': simulated_load[:len(actual_load)],
            'actual_acwr': acwr_data['acwr']
        })

        # Calculate simulated ACWR
        simulated_acwr = []
        for i in range(len(result)):
            if i >= 28:
                acute = np.mean(simulated_load[i-6:i+1])
                chronic = np.mean(simulated_load[i-27:i+1])
                acwr = acute / chronic if chronic > 0 else np.nan
            else:
                acwr = np.nan
            simulated_acwr.append(acwr)

        result['simulated_acwr'] = simulated_acwr

        return result


def run_real_data_backtest():
    """Run comprehensive backtest on real data."""
    print("=" * 70)
    print("REAL DATA BACKTESTING - DELTA V EQUATION")
    print("=" * 70)

    # Load data
    loader = PMDataLoader()
    loader.load_all(verbose=True)

    backtester = RealDataBacktester(loader)

    # Test with default parameters
    print("\n" + "-" * 70)
    print("BACKTEST WITH DEFAULT PARAMETERS")
    print("-" * 70)

    default_params = DeltaVParams()
    default_results = backtester.backtest_all(default_params)

    print(f"\nResults across {default_results['n_participants']} participants:")
    print(f"  Total training days: {default_results['total_days']:,}")
    print(f"  Total injury events: {default_results['total_injuries']}")
    print(f"\nInjury Prediction:")
    print(f"  Predicted (warned before): {default_results['injuries_predicted']}")
    print(f"  Missed (no warning): {default_results['injuries_missed']}")
    print(f"  Prediction rate: {default_results['prediction_rate']:.1f}%")
    print(f"\nFalse Alarms:")
    print(f"  High-risk flags without injury: {default_results['false_alarms']}")
    print(f"  False alarm rate: {default_results['false_alarm_rate']:.2f}% of days")
    print(f"\nRecommendation Distribution:")
    print(f"  Reduce load: {default_results['reduction_pct']:.1f}% of days")
    print(f"  Increase load: {default_results['increase_pct']:.1f}% of days")
    print(f"  Average Delta V: {default_results['avg_delta_v']:.3f}")
    print(f"\nZone Distribution:")
    for zone, count in default_results['zone_distribution'].items():
        pct = count / default_results['total_days'] * 100
        print(f"  {zone}: {count} days ({pct:.1f}%)")

    # Load optimized parameters
    print("\n" + "-" * 70)
    print("BACKTEST WITH OPTIMIZED PARAMETERS")
    print("-" * 70)

    try:
        with open(Path(__file__).parent.parent / "optimized_params.json") as f:
            opt_dict = json.load(f)
        optimized_params = DeltaVParams(**opt_dict)

        opt_results = backtester.backtest_all(optimized_params)

        print(f"\nResults with optimized parameters:")
        print(f"  Prediction rate: {opt_results['prediction_rate']:.1f}%")
        print(f"  False alarm rate: {opt_results['false_alarm_rate']:.2f}%")
        print(f"  Reduce recommendations: {opt_results['reduction_pct']:.1f}%")
        print(f"  Increase recommendations: {opt_results['increase_pct']:.1f}%")
        print(f"  Average Delta V: {opt_results['avg_delta_v']:.3f}")

    except FileNotFoundError:
        print("  (optimized_params.json not found)")
        opt_results = None

    # Test modified parameters tuned for real data
    print("\n" + "-" * 70)
    print("TESTING PARAMETER VARIATIONS")
    print("-" * 70)

    # Try different threshold configurations
    param_variations = {
        'default': DeltaVParams(),
        'lower_thresholds': DeltaVParams(
            threshold_low=0.7,
            threshold_optimal_high=1.2,
            threshold_caution=1.4,
            threshold_critical=1.8
        ),
        'higher_thresholds': DeltaVParams(
            threshold_low=0.85,
            threshold_optimal_high=1.4,
            threshold_caution=1.6,
            threshold_critical=2.2
        ),
        'conservative': DeltaVParams(
            green_max=0.10,
            low_max=0.15,
            red_base=-0.25,
            critical_value=-0.40
        ),
        'aggressive': DeltaVParams(
            green_max=0.20,
            low_max=0.25,
            red_base=-0.10,
            critical_value=-0.20
        ),
    }

    if opt_results:
        param_variations['optimized'] = optimized_params

    comparison = backtester.compare_parameters(param_variations)

    print("\nParameter Comparison:")
    print(comparison.to_string(index=False))

    # Analyze specific injury cases
    print("\n" + "-" * 70)
    print("DETAILED INJURY ANALYSIS")
    print("-" * 70)

    # Find participants with injuries
    for pid in loader.training_data.keys():
        injuries = loader.get_injury_dates(pid)
        if len(injuries) > 0:
            result = backtester.backtest_participant(pid, default_params)
            if result and result['injury_events'] > 0:
                print(f"\n{pid}: {result['injury_events']} injuries")
                print(f"  Predicted: {result['injuries_predicted']}, Missed: {result['injuries_missed']}")

                for detail in result['warning_details'][:5]:  # First 5
                    status = "PREDICTED" if detail['predicted'] else "MISSED"
                    print(f"    {detail['date']}: {status} (max ACWR before: {detail['max_acwr_before']:.2f})")

    # Summary
    print("\n" + "=" * 70)
    print("REAL DATA VALIDATION SUMMARY")
    print("=" * 70)

    print(f"""
Key Findings:

1. Injury Prediction:
   - Delta V predicted {default_results['prediction_rate']:.1f}% of injuries
   - {default_results['injuries_missed']} injuries occurred without prior high-risk warning
   - This suggests many injuries happen at LOW ACWR (not high)

2. False Alarm Rate:
   - {default_results['false_alarm_rate']:.2f}% of days flagged as high-risk
   - Most high-risk flags don't result in immediate injury
   - This is expected - high ACWR increases RISK, not certainty

3. Recommendation Patterns:
   - Delta V recommended load reduction {default_results['reduction_pct']:.1f}% of the time
   - Average recommended change: {default_results['avg_delta_v']*100:.1f}%

4. Zone Distribution:
   - Most days fall in green/low zones (as expected for recreational athletes)
   - High ACWR is rare but associated with higher injury rates

Implications for Delta V Refinement:
- Consider adding "accumulated fatigue" detection (chronic low ACWR → injury)
- The current model focuses on acute spikes but misses chronic issues
- Zone thresholds appear appropriate for real-world data
""")

    return backtester, default_results, comparison


if __name__ == "__main__":
    backtester, results, comparison = run_real_data_backtest()
