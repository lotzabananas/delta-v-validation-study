"""
Report generation utilities for Delta V backtesting.

Generates summary statistics, comparison reports, and formatted output.
"""

from typing import List, Dict, Any, Optional
from dataclasses import asdict
import numpy as np
from datetime import datetime

import sys
sys.path.insert(0, '/Users/timmac/Desktop/Delta V backtesting')

from simulation.engine import SimulationResult, aggregate_results
from core.delta_v import DeltaVParams, format_delta_v_equation


def generate_backtest_report(
    results: List[SimulationResult],
    params: Optional[DeltaVParams] = None,
    title: str = "Delta V Backtesting Report"
) -> str:
    """
    Generate comprehensive text report from simulation results.

    Args:
        results: Simulation results
        params: Parameters used (for documentation)
        title: Report title

    Returns:
        Formatted report string
    """
    agg = aggregate_results(results)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""
{'='*70}
{title}
{'='*70}
Generated: {timestamp}
Simulations: {len(results)}
Weeks per simulation: {len(results[0].weeks) if results else 0}

VOLUME PROGRESSION
------------------
Initial volume (mean):     {np.mean([r.initial_volume for r in results]):>8.1f} min/week
Final volume (mean):       {np.mean([r.final_volume for r in results]):>8.1f} min/week
Growth ratio (mean):       {agg['mean_growth_ratio']:>8.2f}x
Growth ratio (std):        {agg['std_growth_ratio']:>8.2f}
Growth ratio (range):      {agg['min_growth_ratio']:.2f} - {agg['max_growth_ratio']:.2f}
Target reached:            {agg['pct_target_reached']:>8.1f}%

RISK METRICS
------------
Total risk events:         {agg['total_risk_events']:>8d}
Mean risk events/runner:   {agg['mean_risk_events']:>8.2f}
Risk event rate:           {agg['risk_event_rate']:>8.1f}%
Injury proxy triggered:    {agg['pct_with_injury_proxy']:>8.1f}%

STABILITY
---------
Volume change std (mean):  {agg['mean_volume_std']:>8.3f}

"""

    # Per-runner breakdown
    report += """
PER-RUNNER BREAKDOWN
--------------------
"""
    report += f"{'Runner':<20} {'Initial':>8} {'Final':>8} {'Growth':>7} {'Risk':>5} {'Injury':>7}\n"
    report += "-" * 70 + "\n"

    for r in results:
        injury_flag = "YES" if r.injury_proxy_triggered else "no"
        report += (f"{r.profile_name[:20]:<20} "
                  f"{r.initial_volume:>8.0f} "
                  f"{r.final_volume:>8.0f} "
                  f"{r.volume_growth_ratio:>6.2f}x "
                  f"{r.total_risk_events:>5d} "
                  f"{injury_flag:>7}\n")

    # Parameters if provided
    if params:
        report += f"""
DELTA V PARAMETERS
------------------
Green Zone (optimal):
  Base: {params.green_base:.3f}, Range: [{params.green_min:.3f}, {params.green_max:.3f}]

Low Zone (under-training):
  Base: {params.low_base:.3f}, Range: [{params.low_min:.3f}, {params.low_max:.3f}]

Caution Zone: {params.caution_value:.3f}

Red Zone (danger):
  Base: {params.red_base:.3f}, Range: [{params.red_min:.3f}, {params.red_max:.3f}]

Critical Zone: {params.critical_value:.3f}

ACWR Thresholds:
  Low: {params.threshold_low:.2f}
  Optimal High: {params.threshold_optimal_high:.2f}
  Caution: {params.threshold_caution:.2f}
  Critical: {params.threshold_critical:.2f}
"""

    report += "\n" + "=" * 70 + "\n"
    return report


def generate_comparison_report(
    baseline_results: List[SimulationResult],
    optimized_results: List[SimulationResult],
    baseline_params: DeltaVParams,
    optimized_params: DeltaVParams,
    title: str = "Baseline vs Optimized Comparison"
) -> str:
    """
    Generate comparison report between baseline and optimized parameters.

    Args:
        baseline_results: Results with baseline params
        optimized_results: Results with optimized params
        baseline_params: Baseline parameters
        optimized_params: Optimized parameters
        title: Report title

    Returns:
        Formatted comparison report
    """
    baseline_agg = aggregate_results(baseline_results)
    optimized_agg = aggregate_results(optimized_results)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def delta_pct(baseline, optimized):
        if baseline == 0:
            return 0
        return ((optimized - baseline) / baseline) * 100

    report = f"""
{'='*70}
{title}
{'='*70}
Generated: {timestamp}

COMPARISON SUMMARY
------------------
{'Metric':<30} {'Baseline':>12} {'Optimized':>12} {'Change':>12}
{'-'*70}
Growth Ratio (mean)          {baseline_agg['mean_growth_ratio']:>12.2f} {optimized_agg['mean_growth_ratio']:>12.2f} {delta_pct(baseline_agg['mean_growth_ratio'], optimized_agg['mean_growth_ratio']):>+11.1f}%
Target Reached (%)           {baseline_agg['pct_target_reached']:>12.1f} {optimized_agg['pct_target_reached']:>12.1f} {optimized_agg['pct_target_reached'] - baseline_agg['pct_target_reached']:>+11.1f}
Risk Events (mean)           {baseline_agg['mean_risk_events']:>12.2f} {optimized_agg['mean_risk_events']:>12.2f} {optimized_agg['mean_risk_events'] - baseline_agg['mean_risk_events']:>+11.2f}
Risk Event Rate (%)          {baseline_agg['risk_event_rate']:>12.1f} {optimized_agg['risk_event_rate']:>12.1f} {optimized_agg['risk_event_rate'] - baseline_agg['risk_event_rate']:>+11.1f}
Injury Proxy (%)             {baseline_agg['pct_with_injury_proxy']:>12.1f} {optimized_agg['pct_with_injury_proxy']:>12.1f} {optimized_agg['pct_with_injury_proxy'] - baseline_agg['pct_with_injury_proxy']:>+11.1f}
Volume Change Std            {baseline_agg['mean_volume_std']:>12.3f} {optimized_agg['mean_volume_std']:>12.3f} {delta_pct(baseline_agg['mean_volume_std'], optimized_agg['mean_volume_std']):>+11.1f}%

PARAMETER CHANGES
-----------------
{'Parameter':<25} {'Baseline':>12} {'Optimized':>12} {'Change':>12}
{'-'*65}
"""

    # Compare parameters
    baseline_dict = asdict(baseline_params)
    optimized_dict = asdict(optimized_params)

    for key in ['green_base', 'green_max', 'low_base', 'low_max',
                'threshold_low', 'threshold_optimal_high',
                'red_base', 'critical_value']:
        b_val = baseline_dict[key]
        o_val = optimized_dict[key]
        change = o_val - b_val
        report += f"{key:<25} {b_val:>12.4f} {o_val:>12.4f} {change:>+12.4f}\n"

    report += f"""

INTERPRETATION
--------------
"""
    # Add interpretation based on results
    growth_improved = optimized_agg['mean_growth_ratio'] > baseline_agg['mean_growth_ratio']
    risk_reduced = optimized_agg['mean_risk_events'] < baseline_agg['mean_risk_events']

    if growth_improved and risk_reduced:
        report += "The optimized parameters show improvements in BOTH volume growth AND risk reduction.\n"
        report += "This represents a Pareto improvement over the baseline.\n"
    elif growth_improved:
        report += "The optimized parameters improve volume growth but may have higher risk.\n"
        report += "Consider whether the growth-risk tradeoff is acceptable.\n"
    elif risk_reduced:
        report += "The optimized parameters reduce risk but may sacrifice some growth.\n"
        report += "This may be appropriate for injury-prone populations.\n"
    else:
        report += "The optimized parameters do not show clear improvements.\n"
        report += "Consider adjusting optimization weights or constraints.\n"

    report += "\n" + "=" * 70 + "\n"
    return report


def generate_latex_equation(params: DeltaVParams) -> str:
    """
    Generate LaTeX representation of Delta V equation with current parameters.

    Args:
        params: Current parameters

    Returns:
        LaTeX string
    """
    return format_delta_v_equation(params)


def export_results_csv(
    results: List[SimulationResult],
    filepath: str
) -> None:
    """
    Export simulation results to CSV.

    Args:
        results: Simulation results
        filepath: Output file path
    """
    import csv

    headers = [
        'profile_id', 'profile_name', 'initial_volume', 'final_volume',
        'growth_ratio', 'target_reached', 'total_risk_events',
        'max_consecutive_risk', 'injury_triggered', 'volume_change_std'
    ]

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for r in results:
            writer.writerow({
                'profile_id': r.profile_id,
                'profile_name': r.profile_name,
                'initial_volume': r.initial_volume,
                'final_volume': r.final_volume,
                'growth_ratio': r.volume_growth_ratio,
                'target_reached': r.target_volume_reached,
                'total_risk_events': r.total_risk_events,
                'max_consecutive_risk': r.max_consecutive_risk,
                'injury_triggered': r.injury_proxy_triggered,
                'volume_change_std': r.volume_change_std,
            })


def export_weekly_data_csv(
    results: List[SimulationResult],
    filepath: str
) -> None:
    """
    Export weekly data from all simulations to CSV.

    Args:
        results: Simulation results
        filepath: Output file path
    """
    import csv

    headers = [
        'profile_id', 'week', 'target_volume', 'actual_volume',
        'weekly_trimp', 'acwr', 'acwr_zone', 'delta_v', 'flagged'
    ]

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for r in results:
            for w in r.weeks:
                writer.writerow({
                    'profile_id': r.profile_id,
                    'week': w.week,
                    'target_volume': w.target_volume,
                    'actual_volume': w.actual_volume,
                    'weekly_trimp': w.weekly_trimp,
                    'acwr': w.acwr,
                    'acwr_zone': w.acwr_zone,
                    'delta_v': w.delta_v,
                    'flagged': w.flagged,
                })


if __name__ == '__main__':
    print("Testing report generation...")

    from data.synthetic import generate_runner_profiles
    from core.delta_v import DeltaVParams
    from simulation.engine import SimulationEngine

    # Generate test data
    profiles = generate_runner_profiles(10, seed=42)
    params = DeltaVParams()
    engine = SimulationEngine(params)
    results = engine.run_batch(profiles, num_weeks=12, seed=42)

    # Generate report
    report = generate_backtest_report(results, params)
    print(report)

    # Export to CSV
    export_results_csv(results, '/Users/timmac/Desktop/Delta V backtesting/test_results.csv')
    print("\nExported to test_results.csv")

    export_weekly_data_csv(results, '/Users/timmac/Desktop/Delta V backtesting/test_weekly.csv')
    print("Exported to test_weekly.csv")

    print("\nAll report tests passed!")
