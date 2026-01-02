"""
Experiment 004: Base vs Enhanced Delta V Model Comparison
==========================================================

Compare the base Delta V model (ACWR zones only) with the enhanced model
(adding monotony, strain, wellness factors).

Author: Claude (AI Research Assistant)
Date: 2026-01-02
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
import json

from data.runners_loader import RunnersDataLoader
from core.delta_v import DeltaVParams, calculate_delta_v, classify_acwr_zone
from core.delta_v_enhanced import EnhancedDeltaVParams, calculate_enhanced_delta_v, AthleteState


def run_comparison():
    """Compare base vs enhanced model."""
    print("=" * 80)
    print("EXPERIMENT 004: BASE VS ENHANCED MODEL COMPARISON")
    print("=" * 80)

    # Load data
    loader = RunnersDataLoader()
    data = loader.load(verbose=True)

    # Initialize parameters
    base_params = DeltaVParams()
    enhanced_params = EnhancedDeltaVParams()

    # Track recommendations
    results = {
        'base': {'warnings': 0, 'reductions': 0, 'increases': 0, 'days': 0},
        'enhanced': {'warnings': 0, 'reductions': 0, 'increases': 0, 'days': 0},
    }

    # Pre-injury detection (7-day lookback)
    pre_injury_detection = {
        'base': {'detected': 0, 'missed': 0},
        'enhanced': {'detected': 0, 'missed': 0}
    }

    print("\nProcessing athletes...")

    for athlete_id in data['athlete_id'].unique():
        athlete_df = data[data['athlete_id'] == athlete_id].sort_values('date_index')
        athlete_df = athlete_df.reset_index(drop=True)

        # Initialize enhanced state
        state = AthleteState()

        # Find injury starts
        injury_starts = []
        for i, row in athlete_df.iterrows():
            if i == 0:
                continue
            if row['injury'] == 1 and athlete_df.iloc[i-1]['injury'] == 0:
                injury_starts.append(i)

        # Process each day
        for i, row in athlete_df.iterrows():
            acwr = row['acwr']
            load = row['load']

            if pd.isna(acwr):
                continue

            results['base']['days'] += 1
            results['enhanced']['days'] += 1

            # Update state
            state.recent_loads.append(load)
            if len(state.recent_loads) > 7:
                state.recent_loads = state.recent_loads[-7:]

            if load > 0:
                state.consecutive_training_days += 1
            else:
                state.consecutive_training_days = 0

            if acwr < 0.8:
                state.consecutive_low_acwr_days += 1
            else:
                state.consecutive_low_acwr_days = 0

            if acwr > 1.5:
                state.consecutive_high_acwr_days += 1
            else:
                state.consecutive_high_acwr_days = 0

            # Base model
            base_dv, base_zone, base_flag = calculate_delta_v(acwr, base_params)

            if base_dv < -0.05:
                results['base']['reductions'] += 1
            elif base_dv > 0.05:
                results['base']['increases'] += 1

            if base_flag or base_zone in ['critical', 'danger', 'red']:
                results['base']['warnings'] += 1

            # Enhanced model
            enh_dv, enh_zone, enh_flag, meta = calculate_enhanced_delta_v(
                acwr, state, enhanced_params
            )

            if enh_dv < -0.05:
                results['enhanced']['reductions'] += 1
            elif enh_dv > 0.05:
                results['enhanced']['increases'] += 1

            if enh_flag or enh_zone in ['critical', 'danger', 'red', 'strain_critical',
                                        'monotony_critical', 'deload', 'forced_rest']:
                results['enhanced']['warnings'] += 1

            # Check if this day is in pre-injury window
            for inj_idx in injury_starts:
                if inj_idx - 7 <= i < inj_idx:  # 7 days before injury
                    if base_flag or base_zone in ['critical', 'danger', 'red']:
                        pre_injury_detection['base']['detected'] += 1
                    else:
                        pre_injury_detection['base']['missed'] += 1

                    if enh_flag or enh_zone not in ['optimal', 'low', 'caution']:
                        pre_injury_detection['enhanced']['detected'] += 1
                    else:
                        pre_injury_detection['enhanced']['missed'] += 1
                    break

    # Calculate metrics
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(f"\n{'Metric':<35} {'Base Model':>15} {'Enhanced Model':>15}")
    print("-" * 70)

    for metric in ['days', 'warnings', 'reductions', 'increases']:
        base_val = results['base'][metric]
        enh_val = results['enhanced'][metric]
        print(f"{metric.capitalize():<35} {base_val:>15,} {enh_val:>15,}")

    # Warning rates
    base_warn_rate = results['base']['warnings'] / results['base']['days'] * 100
    enh_warn_rate = results['enhanced']['warnings'] / results['enhanced']['days'] * 100

    print(f"\n{'Warning rate (%)':<35} {base_warn_rate:>15.2f} {enh_warn_rate:>15.2f}")

    # Pre-injury detection
    print("\n" + "-" * 70)
    print("PRE-INJURY DETECTION (7-day window)")
    print("-" * 70)

    base_detected = pre_injury_detection['base']['detected']
    base_missed = pre_injury_detection['base']['missed']
    base_rate = base_detected / (base_detected + base_missed) * 100 if (base_detected + base_missed) > 0 else 0

    enh_detected = pre_injury_detection['enhanced']['detected']
    enh_missed = pre_injury_detection['enhanced']['missed']
    enh_rate = enh_detected / (enh_detected + enh_missed) * 100 if (enh_detected + enh_missed) > 0 else 0

    print(f"\n{'Metric':<35} {'Base Model':>15} {'Enhanced Model':>15}")
    print("-" * 70)
    print(f"{'Days flagged in pre-injury window':<35} {base_detected:>15,} {enh_detected:>15,}")
    print(f"{'Days missed in pre-injury window':<35} {base_missed:>15,} {enh_missed:>15,}")
    print(f"{'Detection rate (%)':<35} {base_rate:>15.1f} {enh_rate:>15.1f}")

    improvement = (enh_rate - base_rate) / base_rate * 100 if base_rate > 0 else 0
    print(f"\n{'Improvement':<35} {improvement:>15.1f}%")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"""
Enhanced model generates {enh_warn_rate/base_warn_rate:.1f}x more warnings than base model.

Pre-injury detection:
  - Base model: {base_rate:.1f}% of pre-injury days flagged
  - Enhanced model: {enh_rate:.1f}% of pre-injury days flagged

The enhanced model's additional features (monotony, strain, wellness,
deload scheduling) provide {improvement:+.1f}% improvement in detecting
high-risk periods before injuries occur.
""")

    return {
        'results': results,
        'pre_injury_detection': pre_injury_detection,
        'base_warn_rate': base_warn_rate,
        'enhanced_warn_rate': enh_warn_rate,
        'base_detection_rate': base_rate,
        'enhanced_detection_rate': enh_rate,
        'improvement_pct': improvement
    }


if __name__ == "__main__":
    results = run_comparison()
