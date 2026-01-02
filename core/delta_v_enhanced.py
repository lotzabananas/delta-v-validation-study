"""
Enhanced Delta V Equation with Fatigue Accumulation
=====================================================

Based on real-world validation findings:
1. Only 15% of injuries occur at high ACWR (>1.5)
2. Most injuries occur at low/normal ACWR
3. Chronic fatigue accumulation is a major factor
4. Wellness indicators (HRV, sleep, soreness) matter

This enhanced model adds:
- Monotony detection (lack of variation = risk)
- Chronic fatigue accumulation tracking
- Wellness-based M factor modulation
- Deload triggers for accumulated stress
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.delta_v import DeltaVParams, calculate_delta_v as base_calculate_delta_v


@dataclass
class EnhancedDeltaVParams(DeltaVParams):
    """
    Extended parameters including fatigue and wellness factors.
    """

    # Monotony parameters (Foster's Training Monotony)
    # Monotony = Mean daily load / SD of daily load
    # High monotony (>2.0) with high strain increases injury risk
    monotony_warning_threshold: float = 2.0
    monotony_critical_threshold: float = 2.5
    monotony_reduction: float = -0.10  # Reduce when monotony is high

    # Strain parameters (Foster's Training Strain)
    # Strain = Weekly load × Monotony
    strain_warning_threshold: float = 4000  # sRPE units
    strain_critical_threshold: float = 6000

    # Accumulated fatigue detection
    # If athlete is at low ACWR for extended periods, they may need forced rest
    low_acwr_consecutive_days_warning: int = 14  # 2 weeks at low ACWR
    low_acwr_rest_recommendation: float = -0.20  # Force a reduction

    # Wellness M-factor multipliers
    hrv_low_multiplier: float = 0.7      # Scale recommendations when HRV is low
    sleep_poor_multiplier: float = 0.8    # Scale when sleep is poor
    soreness_high_multiplier: float = 0.6 # Scale when soreness is high
    fatigue_high_multiplier: float = 0.7  # Scale when fatigue is high

    # Deload triggers
    weeks_between_deloads: int = 4   # Suggest deload every N weeks
    deload_reduction: float = -0.40  # 40% reduction during deload week

    # Recovery pattern detection
    no_rest_days_warning: int = 10   # Days without rest day
    forced_rest_threshold: int = 14  # Force rest after this many days


@dataclass
class AthleteState:
    """
    Tracks athlete state for enhanced decision making.
    """
    consecutive_training_days: int = 0
    consecutive_low_acwr_days: int = 0
    consecutive_high_acwr_days: int = 0
    weeks_since_deload: int = 0
    recent_loads: List[float] = field(default_factory=list)  # Last 7 days

    # Wellness (0-10 scale, higher = better for HRV/sleep, worse for fatigue/soreness)
    hrv_status: float = 5.0
    sleep_quality: float = 5.0
    fatigue_level: float = 5.0
    soreness_level: float = 5.0

    def calculate_monotony(self) -> float:
        """Calculate training monotony from recent loads."""
        if len(self.recent_loads) < 7:
            return 1.0  # Default to neutral
        loads = self.recent_loads[-7:]
        mean_load = np.mean(loads)
        std_load = np.std(loads)
        if std_load < 1e-6:
            return 5.0  # Very high monotony if no variation
        return mean_load / std_load

    def calculate_strain(self) -> float:
        """Calculate training strain (load × monotony)."""
        if len(self.recent_loads) < 7:
            return 0
        weekly_load = sum(self.recent_loads[-7:])
        monotony = self.calculate_monotony()
        return weekly_load * monotony

    def calculate_wellness_multiplier(self, params: EnhancedDeltaVParams) -> float:
        """
        Calculate wellness-based multiplier for recommendations.

        Returns a value 0-1 that scales the recommended change.
        Lower values = more conservative recommendations.
        """
        multiplier = 1.0

        # HRV impact (assume 0-10 scale, <4 is low)
        if self.hrv_status < 4:
            multiplier *= params.hrv_low_multiplier

        # Sleep impact (<4 is poor)
        if self.sleep_quality < 4:
            multiplier *= params.sleep_poor_multiplier

        # Fatigue impact (>6 is high fatigue)
        if self.fatigue_level > 6:
            multiplier *= params.fatigue_high_multiplier

        # Soreness impact (>6 is high soreness)
        if self.soreness_level > 6:
            multiplier *= params.soreness_high_multiplier

        return multiplier


def calculate_enhanced_delta_v(
    acwr: float,
    state: AthleteState,
    params: EnhancedDeltaVParams = None,
) -> Tuple[float, str, bool, Dict[str, Any]]:
    """
    Calculate enhanced Delta V considering fatigue, wellness, and monotony.

    Args:
        acwr: Current ACWR
        state: Current athlete state
        params: Enhanced parameters

    Returns:
        Tuple of (delta_v, zone, flag_for_review, metadata)
    """
    if params is None:
        params = EnhancedDeltaVParams()

    metadata = {
        'base_recommendation': None,
        'adjustments': [],
        'wellness_multiplier': 1.0,
        'monotony': None,
        'strain': None,
    }

    # Calculate base recommendation from original Delta V
    base_delta_v, base_zone, base_flag = base_calculate_delta_v(
        acwr, params, state.consecutive_high_acwr_days
    )
    metadata['base_recommendation'] = base_delta_v
    metadata['base_zone'] = base_zone

    delta_v = base_delta_v
    zone = base_zone
    flag = base_flag

    # ═══════════════════════════════════════════════════════════════════════
    # CHECK 1: Deload trigger
    # ═══════════════════════════════════════════════════════════════════════
    if state.weeks_since_deload >= params.weeks_between_deloads:
        metadata['adjustments'].append(f"Deload recommended (week {state.weeks_since_deload})")
        delta_v = params.deload_reduction
        zone = 'deload'
        flag = True
        return delta_v, zone, flag, metadata

    # ═══════════════════════════════════════════════════════════════════════
    # CHECK 2: Forced rest if no rest days
    # ═══════════════════════════════════════════════════════════════════════
    if state.consecutive_training_days >= params.forced_rest_threshold:
        metadata['adjustments'].append(f"Forced rest (training {state.consecutive_training_days} consecutive days)")
        delta_v = -1.0  # Full rest day recommended
        zone = 'forced_rest'
        flag = True
        return delta_v, zone, flag, metadata

    # ═══════════════════════════════════════════════════════════════════════
    # CHECK 3: Monotony warning
    # ═══════════════════════════════════════════════════════════════════════
    monotony = state.calculate_monotony()
    metadata['monotony'] = monotony

    if monotony >= params.monotony_critical_threshold:
        metadata['adjustments'].append(f"Critical monotony ({monotony:.2f})")
        delta_v = min(delta_v, params.monotony_reduction * 1.5)
        zone = 'monotony_critical'
        flag = True
    elif monotony >= params.monotony_warning_threshold:
        metadata['adjustments'].append(f"High monotony ({monotony:.2f})")
        # Reduce any increase recommendation
        if delta_v > 0:
            delta_v = delta_v * 0.5
            metadata['adjustments'].append("Halved increase due to monotony")

    # ═══════════════════════════════════════════════════════════════════════
    # CHECK 4: Strain warning
    # ═══════════════════════════════════════════════════════════════════════
    strain = state.calculate_strain()
    metadata['strain'] = strain

    if strain >= params.strain_critical_threshold:
        metadata['adjustments'].append(f"Critical strain ({strain:.0f})")
        delta_v = min(delta_v, -0.20)
        zone = 'strain_critical'
        flag = True
    elif strain >= params.strain_warning_threshold:
        metadata['adjustments'].append(f"High strain ({strain:.0f})")
        if delta_v > 0:
            delta_v = delta_v * 0.5
            metadata['adjustments'].append("Halved increase due to strain")

    # ═══════════════════════════════════════════════════════════════════════
    # CHECK 5: Extended low ACWR (chronic underload but potential fatigue)
    # ═══════════════════════════════════════════════════════════════════════
    if state.consecutive_low_acwr_days >= params.low_acwr_consecutive_days_warning:
        # This is counter-intuitive: low ACWR for extended period
        # might indicate chronic fatigue masked by reduced training
        metadata['adjustments'].append(f"Extended low ACWR ({state.consecutive_low_acwr_days} days)")
        # Don't aggressively increase - maintain or slight reduction
        if delta_v > 0.05:
            delta_v = 0.05  # Cap increase at 5%
            metadata['adjustments'].append("Capped increase due to extended low period")

    # ═══════════════════════════════════════════════════════════════════════
    # CHECK 6: Wellness multiplier
    # ═══════════════════════════════════════════════════════════════════════
    wellness_mult = state.calculate_wellness_multiplier(params)
    metadata['wellness_multiplier'] = wellness_mult

    if wellness_mult < 1.0:
        if delta_v > 0:
            # Scale back increases when wellness is poor
            delta_v = delta_v * wellness_mult
            metadata['adjustments'].append(f"Wellness scaling ({wellness_mult:.2f})")
        elif delta_v < 0:
            # Increase the magnitude of reductions when wellness is poor
            delta_v = delta_v / wellness_mult
            metadata['adjustments'].append(f"Enhanced reduction due to wellness ({wellness_mult:.2f})")

    # ═══════════════════════════════════════════════════════════════════════
    # CHECK 7: No rest days warning
    # ═══════════════════════════════════════════════════════════════════════
    if state.consecutive_training_days >= params.no_rest_days_warning:
        metadata['adjustments'].append(f"Warning: {state.consecutive_training_days} days without rest")
        flag = True

    return delta_v, zone, flag, metadata


def run_enhanced_validation():
    """
    Run validation comparing base vs enhanced Delta V on real data.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from data.pmdata_loader import PMDataLoader
    import pandas as pd

    print("=" * 70)
    print("ENHANCED DELTA V VALIDATION")
    print("=" * 70)

    # Load data
    loader = PMDataLoader()
    loader.load_all(verbose=True)

    # For each participant with injuries, test enhanced model
    print("\n" + "-" * 70)
    print("COMPARING BASE VS ENHANCED MODEL")
    print("-" * 70)

    enhanced_params = EnhancedDeltaVParams()

    for pid in loader.training_data.keys():
        injuries = loader.get_injury_dates(pid)
        if len(injuries) == 0:
            continue

        acwr_data = loader.get_acwr_series(pid)
        if len(acwr_data) == 0:
            continue

        # Get wellness data if available
        wellness = loader.wellness_data.get(pid, pd.DataFrame())

        # Simulate with enhanced model
        state = AthleteState()
        enhanced_warnings = 0
        base_warnings = 0

        for i, row in acwr_data.iterrows():
            if pd.isna(row['acwr']):
                continue

            # Update state
            load = row['load']
            state.recent_loads.append(load)
            if len(state.recent_loads) > 7:
                state.recent_loads = state.recent_loads[-7:]

            if load > 0:
                state.consecutive_training_days += 1
            else:
                state.consecutive_training_days = 0

            if row['acwr'] < 0.8:
                state.consecutive_low_acwr_days += 1
            else:
                state.consecutive_low_acwr_days = 0

            if row['acwr'] > 1.5:
                state.consecutive_high_acwr_days += 1
            else:
                state.consecutive_high_acwr_days = 0

            # Get recommendations
            base_dv, base_zone, base_flag = base_calculate_delta_v(row['acwr'])
            enh_dv, enh_zone, enh_flag, meta = calculate_enhanced_delta_v(
                row['acwr'], state, enhanced_params
            )

            if base_zone in ['red', 'critical', 'danger']:
                base_warnings += 1
            if enh_flag or enh_zone in ['red', 'critical', 'danger', 'monotony_critical', 'strain_critical']:
                enhanced_warnings += 1

        n_injuries = len(injuries)
        print(f"\n{pid}: {n_injuries} injuries")
        print(f"  Base model warnings: {base_warnings}")
        print(f"  Enhanced model warnings: {enhanced_warnings}")
        print(f"  Improvement: {enhanced_warnings - base_warnings:+d} additional warnings")

    # Summary
    print("\n" + "=" * 70)
    print("ENHANCED MODEL FEATURES")
    print("=" * 70)
    print("""
The enhanced Delta V model adds:

1. MONOTONY DETECTION
   - Flags when daily loads are too similar (no variation)
   - High monotony with high load = increased injury risk

2. STRAIN TRACKING
   - Strain = Weekly load × Monotony
   - Forces reduction when strain exceeds thresholds

3. EXTENDED LOW ACWR WARNING
   - Flags when athlete is at low ACWR for >2 weeks
   - Prevents aggressive increases during potential overtraining

4. WELLNESS INTEGRATION
   - Scales recommendations based on HRV, sleep, fatigue, soreness
   - Poor wellness = more conservative recommendations

5. DELOAD SCHEDULING
   - Recommends deload every 4 weeks
   - Prevents accumulated fatigue

6. REST DAY ENFORCEMENT
   - Warns after 10+ consecutive training days
   - Forces rest after 14+ days

These additions address the key finding that most injuries occur
at LOW or NORMAL ACWR, not high ACWR.
""")


if __name__ == "__main__":
    run_enhanced_validation()
