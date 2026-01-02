"""
Delta V Equation: Parameterized weekly volume adjustment based on ACWR.

The Delta V equation outputs a percentage change to apply to weekly running volume,
designed to ensure safe progressive overload while minimizing injury risk.

Based on:
- ACWR research from Gabbett (2016)
- Running progression heuristics from Nielsen et al. (2014)
- 10% rule and its nuanced application
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Tuple
import numpy as np


@dataclass
class DeltaVParams:
    """
    Tunable parameters for the Delta V equation.

    All percentage values are expressed as decimals (e.g., 0.15 = 15%).
    Thresholds define ACWR zone boundaries.
    """

    # ═══════════════════════════════════════════════════════════════════════════
    # ACWR THRESHOLDS (zone boundaries)
    # ═══════════════════════════════════════════════════════════════════════════

    threshold_low: float = 0.8          # Below this: under-training zone
    threshold_optimal_high: float = 1.3  # Above this: leaving optimal zone
    threshold_caution: float = 1.5       # Above this: caution zone ends
    threshold_critical: float = 2.0      # Above this: critical zone

    # ═══════════════════════════════════════════════════════════════════════════
    # GREEN ZONE: Optimal training (0.8 <= ACWR < 1.3)
    # ═══════════════════════════════════════════════════════════════════════════
    # In this zone, we can safely increase volume
    # The increase scales inversely with ACWR (lower ACWR = more room to grow)

    green_base: float = 0.20     # Base multiplier for green zone calculation
    green_min: float = 0.05      # Minimum increase in green zone (5%)
    green_max: float = 0.15      # Maximum increase in green zone (15%)

    # ═══════════════════════════════════════════════════════════════════════════
    # LOW ZONE: Under-training (ACWR < 0.8)
    # ═══════════════════════════════════════════════════════════════════════════
    # Chronic load is high relative to acute - athlete can handle more
    # Safe to increase more aggressively

    low_base: float = 0.25       # Base multiplier for low zone
    low_min: float = 0.10        # Minimum increase (10%)
    low_max: float = 0.20        # Maximum increase (20%)

    # ═══════════════════════════════════════════════════════════════════════════
    # CAUTION ZONE: Elevated risk (1.3 <= ACWR < 1.5)
    # ═══════════════════════════════════════════════════════════════════════════
    # Hold steady - no increase, no decrease

    caution_value: float = 0.0   # No change in caution zone

    # ═══════════════════════════════════════════════════════════════════════════
    # RED ZONE: High risk (1.5 <= ACWR < 2.0)
    # ═══════════════════════════════════════════════════════════════════════════
    # Need to reduce volume to let chronic load catch up

    red_base: float = -0.20      # Base multiplier for reduction
    red_min: float = -0.15       # Minimum reduction (less aggressive)
    red_max: float = -0.05       # Maximum reduction (cap on how much we cut)

    # ═══════════════════════════════════════════════════════════════════════════
    # CRITICAL ZONE: Very high risk (ACWR >= 2.0)
    # ═══════════════════════════════════════════════════════════════════════════
    # Immediate significant reduction needed

    critical_value: float = -0.30     # 30% reduction
    critical_flag: bool = True        # Flag for review/deload trigger

    # ═══════════════════════════════════════════════════════════════════════════
    # PERSISTENCE RULES
    # ═══════════════════════════════════════════════════════════════════════════
    # Track consecutive weeks in concerning zones

    persistent_weeks_threshold: int = 2   # Weeks before escalation
    persistent_critical_action: float = -0.30  # Action if persistent >1.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DeltaVParams':
        """Create parameters from dictionary."""
        return cls(**d)

    def validate(self) -> Tuple[bool, str]:
        """Validate parameter constraints."""
        issues = []

        # Threshold ordering
        if not (0 < self.threshold_low < self.threshold_optimal_high
                < self.threshold_caution < self.threshold_critical):
            issues.append("ACWR thresholds must be in ascending order")

        # Green zone bounds
        if not (0 <= self.green_min <= self.green_max <= 0.25):
            issues.append("Green zone: 0 <= min <= max <= 0.25")

        # Low zone bounds
        if not (0 <= self.low_min <= self.low_max <= 0.30):
            issues.append("Low zone: 0 <= min <= max <= 0.30")

        # Red zone bounds (negative values)
        if not (-0.30 <= self.red_base <= 0):
            issues.append("Red base must be in [-0.30, 0]")
        if not (self.red_min <= self.red_max <= 0):
            issues.append("Red zone: min <= max <= 0")

        # Critical value
        if not (-0.50 <= self.critical_value <= -0.10):
            issues.append("Critical value must be in [-0.50, -0.10]")

        if issues:
            return False, "; ".join(issues)
        return True, "Valid"


def calculate_delta_v(
    acwr: float,
    params: Optional[DeltaVParams] = None,
    consecutive_high_acwr_weeks: int = 0
) -> Tuple[float, str, bool]:
    """
    Calculate the Delta V (percentage volume change) based on ACWR.

    The piecewise function determines volume adjustment:
    - Low ACWR (< 0.8): Increase 10-20%
    - Optimal ACWR (0.8-1.3): Increase 5-15%
    - Caution ACWR (1.3-1.5): Maintain (0%)
    - High ACWR (1.5-2.0): Decrease 5-15%
    - Critical ACWR (>= 2.0): Decrease 30%

    Args:
        acwr: Current Acute:Chronic Workload Ratio
        params: DeltaVParams with tunable coefficients (uses defaults if None)
        consecutive_high_acwr_weeks: Number of consecutive weeks with ACWR > 1.5

    Returns:
        Tuple of (delta_v, zone_name, flag_for_review)
        - delta_v: Percentage change as decimal (e.g., 0.10 = +10%)
        - zone_name: Classification of current zone
        - flag_for_review: Whether this needs human review
    """
    if params is None:
        params = DeltaVParams()

    flag_for_review = False

    # ═══════════════════════════════════════════════════════════════════════════
    # CRITICAL ZONE: ACWR >= 2.0
    # ═══════════════════════════════════════════════════════════════════════════
    if acwr >= params.threshold_critical:
        return params.critical_value, 'critical', params.critical_flag

    # ═══════════════════════════════════════════════════════════════════════════
    # PERSISTENCE CHECK: >2 consecutive weeks above 1.5
    # ═══════════════════════════════════════════════════════════════════════════
    if consecutive_high_acwr_weeks >= params.persistent_weeks_threshold:
        return params.persistent_critical_action, 'persistent_high', True

    # ═══════════════════════════════════════════════════════════════════════════
    # RED ZONE: 1.5 <= ACWR < 2.0
    # ═══════════════════════════════════════════════════════════════════════════
    if acwr >= params.threshold_caution:
        # Linear scaling: higher ACWR = more reduction
        # delta_v = base × (ACWR - 1.3)
        raw_delta = params.red_base * (acwr - params.threshold_optimal_high)
        delta_v = min(params.red_max, max(params.red_min, raw_delta))
        return delta_v, 'danger', False

    # ═══════════════════════════════════════════════════════════════════════════
    # CAUTION ZONE: 1.3 <= ACWR < 1.5
    # ═══════════════════════════════════════════════════════════════════════════
    if acwr >= params.threshold_optimal_high:
        return params.caution_value, 'caution', False

    # ═══════════════════════════════════════════════════════════════════════════
    # OPTIMAL (GREEN) ZONE: 0.8 <= ACWR < 1.3
    # ═══════════════════════════════════════════════════════════════════════════
    if acwr >= params.threshold_low:
        # Scale increase inversely with ACWR
        # Higher ACWR within green zone = smaller increase
        # Formula: base × (1 - (ACWR - 0.8) / 0.5)
        # At ACWR = 0.8: factor = 1.0 (full increase)
        # At ACWR = 1.3: factor = 0.0 (minimal increase)
        range_width = params.threshold_optimal_high - params.threshold_low
        normalized_position = (acwr - params.threshold_low) / range_width
        scaling_factor = 1.0 - normalized_position

        raw_delta = params.green_base * scaling_factor
        delta_v = max(params.green_min, min(params.green_max, raw_delta))
        return delta_v, 'optimal', False

    # ═══════════════════════════════════════════════════════════════════════════
    # LOW ZONE: ACWR < 0.8
    # ═══════════════════════════════════════════════════════════════════════════
    # Under-training - can increase more aggressively
    # Formula: base × (0.8 - ACWR + 0.1)
    # Lower ACWR = larger increase
    adjustment = params.threshold_low - acwr + 0.1
    raw_delta = params.low_base * adjustment
    delta_v = max(params.low_min, min(params.low_max, raw_delta))
    return delta_v, 'low', False


def apply_delta_v(current_volume: float, delta_v: float) -> float:
    """
    Apply Delta V to current volume.

    Args:
        current_volume: Current weekly volume (minutes)
        delta_v: Percentage change as decimal

    Returns:
        New weekly volume (minutes)
    """
    new_volume = current_volume * (1 + delta_v)
    # Enforce minimum volume (can't go below ~20 min/week)
    return max(20.0, new_volume)


def get_delta_v_summary(
    acwr: float,
    current_volume: float,
    params: Optional[DeltaVParams] = None,
    consecutive_high_acwr_weeks: int = 0
) -> Dict[str, Any]:
    """
    Get full Delta V calculation summary.

    Args:
        acwr: Current ACWR
        current_volume: Current weekly volume (minutes)
        params: DeltaVParams
        consecutive_high_acwr_weeks: Consecutive weeks with high ACWR

    Returns:
        Dictionary with all calculation details
    """
    delta_v, zone, flag = calculate_delta_v(acwr, params, consecutive_high_acwr_weeks)
    new_volume = apply_delta_v(current_volume, delta_v)

    return {
        'acwr': acwr,
        'zone': zone,
        'delta_v': delta_v,
        'delta_v_percent': f"{delta_v * 100:+.1f}%",
        'current_volume': current_volume,
        'new_volume': new_volume,
        'volume_change': new_volume - current_volume,
        'flag_for_review': flag,
        'consecutive_high_weeks': consecutive_high_acwr_weeks,
    }


def format_delta_v_equation(params: DeltaVParams) -> str:
    """
    Format the Delta V equation as LaTeX for documentation.

    Args:
        params: Current parameters

    Returns:
        LaTeX string representation
    """
    latex = r"""
\Delta V =
\begin{cases}
\max(%.2f, \min(%.2f, %.2f \times (1 - \frac{\text{ACWR} - %.1f}{%.1f}))) & \text{if } %.1f \leq \text{ACWR} < %.1f \\
\max(%.2f, \min(%.2f, %.2f \times (%.1f - \text{ACWR} + 0.1))) & \text{if } \text{ACWR} < %.1f \\
%.2f & \text{if } %.1f \leq \text{ACWR} < %.1f \\
\min(%.2f, \max(%.2f, %.2f \times (\text{ACWR} - %.1f))) & \text{if } %.1f \leq \text{ACWR} < %.1f \\
%.2f \text{ (flag for review)} & \text{if } \text{ACWR} \geq %.1f
\end{cases}
""" % (
        params.green_min, params.green_max, params.green_base,
        params.threshold_low,
        params.threshold_optimal_high - params.threshold_low,
        params.threshold_low, params.threshold_optimal_high,

        params.low_min, params.low_max, params.low_base,
        params.threshold_low, params.threshold_low,

        params.caution_value,
        params.threshold_optimal_high, params.threshold_caution,

        params.red_max, params.red_min, params.red_base,
        params.threshold_optimal_high,
        params.threshold_caution, params.threshold_critical,

        params.critical_value, params.threshold_critical,
    )
    return latex


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETER SEARCH BOUNDS (for optimization)
# ═══════════════════════════════════════════════════════════════════════════════

PARAM_BOUNDS = {
    # Thresholds
    'threshold_low': (0.6, 0.9),
    'threshold_optimal_high': (1.2, 1.4),
    'threshold_caution': (1.4, 1.6),
    'threshold_critical': (1.8, 2.2),

    # Green zone
    'green_base': (0.15, 0.30),
    'green_min': (0.03, 0.08),
    'green_max': (0.12, 0.20),

    # Low zone
    'low_base': (0.20, 0.35),
    'low_min': (0.08, 0.15),
    'low_max': (0.15, 0.25),

    # Caution zone
    'caution_value': (-0.05, 0.05),

    # Red zone
    'red_base': (-0.30, -0.10),
    'red_min': (-0.20, -0.10),
    'red_max': (-0.08, -0.02),

    # Critical zone
    'critical_value': (-0.40, -0.20),
}


if __name__ == '__main__':
    # Quick validation tests
    print("Testing Delta V equation...")

    params = DeltaVParams()

    # Test each zone
    test_cases = [
        (0.5, "low"),
        (0.75, "low"),
        (0.9, "optimal"),
        (1.1, "optimal"),
        (1.35, "caution"),
        (1.45, "caution"),
        (1.6, "danger"),
        (1.8, "danger"),
        (2.1, "critical"),
    ]

    print("\nACWR -> Zone -> Delta V:")
    print("-" * 50)
    for acwr, expected_zone in test_cases:
        result = get_delta_v_summary(acwr, 120, params)
        print(f"ACWR {acwr:.2f}: {result['zone']:12s} -> {result['delta_v_percent']:>7s} "
              f"({result['current_volume']:.0f} -> {result['new_volume']:.0f} min)")
        assert result['zone'] == expected_zone, f"Expected {expected_zone}, got {result['zone']}"

    # Test persistence
    print("\nPersistence test (3 weeks at ACWR 1.6):")
    result = get_delta_v_summary(1.6, 120, params, consecutive_high_acwr_weeks=3)
    print(f"Zone: {result['zone']}, Delta V: {result['delta_v_percent']}, Flag: {result['flag_for_review']}")

    print("\nAll tests passed!")
