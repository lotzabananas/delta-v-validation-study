"""
Core training metrics: TRIMP, EWMA, and ACWR calculations.

Based on:
- Banister (1991): TRIMP formula
- Williams et al. (2017): EWMA-based ACWR
- Gabbett (2016): ACWR injury risk thresholds
"""

import numpy as np
from typing import List, Optional, Tuple
import math


def calculate_delta_hr(hr_avg: float, hr_rest: float, hr_max: float) -> float:
    """
    Calculate heart rate reserve fraction (Delta HR).

    Args:
        hr_avg: Average heart rate during session (bpm)
        hr_rest: Resting heart rate (bpm)
        hr_max: Maximum heart rate (bpm)

    Returns:
        Delta HR as fraction [0, 1]
    """
    if hr_max <= hr_rest:
        raise ValueError(f"hr_max ({hr_max}) must be greater than hr_rest ({hr_rest})")

    delta_hr = (hr_avg - hr_rest) / (hr_max - hr_rest)
    return np.clip(delta_hr, 0.0, 1.0)


def calculate_y_factor(delta_hr: float, gender: str = 'male') -> float:
    """
    Calculate the Y weighting factor for TRIMP.

    Based on Banister's gender-specific exponential weighting.

    Args:
        delta_hr: Heart rate reserve fraction
        gender: 'male' or 'female'

    Returns:
        Y factor (exponential intensity weighting)
    """
    if gender.lower() == 'male':
        # Males: Y = 0.64 * e^(1.92 * deltaHR)
        return 0.64 * math.exp(1.92 * delta_hr)
    elif gender.lower() == 'female':
        # Females: Y = 0.86 * e^(1.67 * deltaHR)
        return 0.86 * math.exp(1.67 * delta_hr)
    else:
        raise ValueError(f"Gender must be 'male' or 'female', got '{gender}'")


def calculate_trimp(
    duration_min: float,
    hr_avg: float,
    hr_rest: float = 60.0,
    hr_max: float = None,
    age: int = None,
    gender: str = 'male'
) -> float:
    """
    Calculate Training Impulse (TRIMP) for a session.

    TRIMP = Duration × ΔHR × Y

    Args:
        duration_min: Session duration in minutes
        hr_avg: Average heart rate during session (bpm)
        hr_rest: Resting heart rate (bpm), default 60
        hr_max: Maximum heart rate (bpm), or None to estimate from age
        age: Age in years (used if hr_max not provided)
        gender: 'male' or 'female'

    Returns:
        TRIMP value (arbitrary units)
    """
    # Estimate hr_max if not provided
    if hr_max is None:
        if age is None:
            raise ValueError("Must provide either hr_max or age")
        hr_max = 220 - age  # Standard formula

    # Calculate components
    delta_hr = calculate_delta_hr(hr_avg, hr_rest, hr_max)
    y_factor = calculate_y_factor(delta_hr, gender)

    # TRIMP = D × ΔHR × Y
    trimp = duration_min * delta_hr * y_factor

    return trimp


def calculate_ewma(
    values: np.ndarray,
    span: int,
    min_periods: int = 1
) -> np.ndarray:
    """
    Calculate Exponentially Weighted Moving Average.

    Uses the formula: EWMA_t = value_t × λ + (1 - λ) × EWMA_{t-1}
    where λ = 2 / (span + 1)

    Args:
        values: Array of daily values (e.g., TRIMP)
        span: Decay span (7 for acute, 28 for chronic)
        min_periods: Minimum number of observations required

    Returns:
        Array of EWMA values
    """
    values = np.asarray(values, dtype=float)
    n = len(values)

    if n == 0:
        return np.array([])

    # Calculate decay factor (λ)
    alpha = 2.0 / (span + 1.0)

    # Initialize output
    ewma = np.zeros(n)
    ewma[0] = values[0]

    # Calculate EWMA iteratively
    for i in range(1, n):
        ewma[i] = alpha * values[i] + (1 - alpha) * ewma[i - 1]

    return ewma


def calculate_acwr(
    trimp_history: np.ndarray,
    acute_span: int = 7,
    chronic_span: int = 28,
    min_chronic_days: int = 14
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Calculate Acute:Chronic Workload Ratio using EWMA method.

    ACWR = Acute EWMA (7-day) / Chronic EWMA (28-day)

    Args:
        trimp_history: Array of daily TRIMP values
        acute_span: Span for acute load (default 7 days)
        chronic_span: Span for chronic load (default 28 days)
        min_chronic_days: Minimum days before ACWR is meaningful

    Returns:
        Tuple of (current_acwr, acute_ewma_array, chronic_ewma_array)
    """
    trimp_history = np.asarray(trimp_history, dtype=float)

    if len(trimp_history) == 0:
        return 1.0, np.array([]), np.array([])

    # Calculate EWMA for acute and chronic
    acute_ewma = calculate_ewma(trimp_history, span=acute_span)
    chronic_ewma = calculate_ewma(trimp_history, span=chronic_span)

    # Avoid division by zero
    # When chronic load is very low, ACWR can spike unrealistically
    # Use a minimum chronic threshold based on typical zone 2 training
    min_chronic = 10.0  # Minimum meaningful chronic load

    if len(trimp_history) < min_chronic_days:
        # Not enough history - return neutral ACWR
        current_acwr = 1.0
    elif chronic_ewma[-1] < min_chronic:
        # Very low chronic load - athlete is under-trained
        # High acute relative to low chronic suggests ramping up
        current_acwr = 0.5 if acute_ewma[-1] < min_chronic else 1.5
    else:
        current_acwr = acute_ewma[-1] / chronic_ewma[-1]

    return current_acwr, acute_ewma, chronic_ewma


def calculate_acwr_series(
    trimp_history: np.ndarray,
    acute_span: int = 7,
    chronic_span: int = 28
) -> np.ndarray:
    """
    Calculate ACWR for entire time series.

    Args:
        trimp_history: Array of daily TRIMP values
        acute_span: Span for acute load
        chronic_span: Span for chronic load

    Returns:
        Array of ACWR values for each day
    """
    trimp_history = np.asarray(trimp_history, dtype=float)

    if len(trimp_history) == 0:
        return np.array([])

    acute_ewma = calculate_ewma(trimp_history, span=acute_span)
    chronic_ewma = calculate_ewma(trimp_history, span=chronic_span)

    # Calculate ACWR with protection against division by zero
    min_chronic = 10.0
    acwr = np.where(
        chronic_ewma < min_chronic,
        1.0,  # Default to neutral when chronic is too low
        acute_ewma / chronic_ewma
    )

    return acwr


def estimate_hr_max(age: int) -> float:
    """Estimate maximum heart rate from age using standard formula."""
    return 220 - age


def estimate_maf_hr(age: int, adjustment: int = 0) -> float:
    """
    Calculate MAF (Maximum Aerobic Function) heart rate.

    MAF HR = 180 - age + adjustment

    Adjustments:
        +5: Experienced athlete, training consistently 2+ years
        0: Regular exerciser, generally healthy
        -5: Recovering from illness, inconsistent training
        -10: Major health issues, very sedentary

    Args:
        age: Age in years
        adjustment: MAF adjustment factor

    Returns:
        MAF heart rate (zone 2 ceiling)
    """
    return 180 - age + adjustment


def calculate_weekly_trimp(
    daily_trimps: np.ndarray,
    week_start_idx: int = 0
) -> float:
    """
    Sum TRIMP values for a week.

    Args:
        daily_trimps: Array of daily TRIMP values
        week_start_idx: Starting index for the week

    Returns:
        Total weekly TRIMP
    """
    week_end_idx = min(week_start_idx + 7, len(daily_trimps))
    return np.sum(daily_trimps[week_start_idx:week_end_idx])


def classify_acwr_zone(acwr: float) -> str:
    """
    Classify ACWR into training zones.

    Zones based on Gabbett (2016):
        - Low: < 0.8 (under-training, can increase load)
        - Optimal: 0.8-1.3 (sweet spot for gains)
        - Caution: 1.3-1.5 (elevated risk, maintain)
        - Danger: 1.5-2.0 (reduce load)
        - Critical: >= 2.0 (significant injury risk)

    Args:
        acwr: Acute:Chronic Workload Ratio

    Returns:
        Zone classification string
    """
    if acwr < 0.8:
        return 'low'
    elif acwr < 1.3:
        return 'optimal'
    elif acwr < 1.5:
        return 'caution'
    elif acwr < 2.0:
        return 'danger'
    else:
        return 'critical'


# Validation helpers for testing

def validate_trimp_range(trimp: float, duration_min: float) -> bool:
    """
    Validate TRIMP is in reasonable range.

    For zone 2 training (moderate intensity), expect roughly:
    - 30 min session: TRIMP ~30-60
    - 60 min session: TRIMP ~60-120
    - 90 min session: TRIMP ~90-180
    """
    if trimp < 0:
        return False

    # Rough bounds: TRIMP should be 0.5-4x duration for typical training
    lower_bound = duration_min * 0.3
    upper_bound = duration_min * 5.0

    return lower_bound <= trimp <= upper_bound


def validate_acwr_range(acwr: float) -> bool:
    """Validate ACWR is in physiologically plausible range."""
    return 0.0 <= acwr <= 5.0


if __name__ == '__main__':
    # Quick validation tests
    print("Testing core metrics...")

    # Test TRIMP calculation
    trimp = calculate_trimp(
        duration_min=60,
        hr_avg=140,
        hr_rest=60,
        hr_max=180,
        gender='male'
    )
    print(f"60 min session, HR 140: TRIMP = {trimp:.2f}")
    assert validate_trimp_range(trimp, 60), "TRIMP out of expected range"

    # Test EWMA
    values = np.array([100, 110, 105, 120, 95, 100, 115])
    ewma_7 = calculate_ewma(values, span=7)
    print(f"7-day EWMA of {values}: {ewma_7}")

    # Test ACWR with 28 days of data
    np.random.seed(42)
    daily_trimp = np.random.normal(50, 10, 35)  # 5 weeks of data
    acwr, acute, chronic = calculate_acwr(daily_trimp)
    print(f"Current ACWR: {acwr:.2f}")
    print(f"Zone: {classify_acwr_zone(acwr)}")

    print("\nAll tests passed!")
