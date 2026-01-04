"""
Frequency Equations: Days per week calculations for running plans.

Based on:
- Fitzgerald, M. (2014). 80/20 Running
- Seiler, S. (2010). Polarized training distribution
- Progressive overload principles

These equations determine optimal running frequency based on volume,
training phase, recovery capacity, and individual factors.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, Tuple
import math

from .maf import ExperienceLevel


class TrainingPhase(Enum):
    """Training phase classification."""
    BASE_BUILDING = "base_building"   # Pure Zone 2, aerobic development
    TRANSITION = "transition"         # Introduction of intensity (90/10)
    EIGHTY_TWENTY = "80_20"           # Full polarized (80/20)
    RACE_SPECIFIC = "race_specific"   # Goal-pace work
    TAPER = "taper"                   # Pre-race reduction
    RECOVERY = "recovery"             # Post-race or injury return


@dataclass
class RecoveryMetrics:
    """
    Recovery quality metrics from wearables/subjective input.

    All values on 0-1 scale where 1 = excellent recovery.
    """
    hrv_score: float = 0.7          # HRV relative to baseline
    sleep_quality: float = 0.7      # Sleep score (0-1)
    sleep_hours: float = 7.0        # Hours of sleep
    fatigue_level: float = 0.3      # Subjective fatigue (0=none, 1=extreme)
    soreness_level: float = 0.3     # Muscle soreness (0=none, 1=extreme)
    stress_level: float = 0.3       # Life stress (0=none, 1=extreme)

    @property
    def recovery_score(self) -> float:
        """
        Calculate composite recovery score.

        Returns value 0-1 where 1 = fully recovered, 0 = exhausted.
        """
        # Weight factors
        weights = {
            'hrv': 0.25,
            'sleep_quality': 0.20,
            'sleep_hours': 0.15,
            'fatigue': 0.15,
            'soreness': 0.15,
            'stress': 0.10,
        }

        # Normalize sleep hours (7-9 hours = 1.0, <5 or >10 = 0.5)
        sleep_norm = min(1.0, max(0.0, (self.sleep_hours - 5) / 4))
        if self.sleep_hours > 9:
            sleep_norm = max(0.5, 1.0 - (self.sleep_hours - 9) / 2)

        # Calculate weighted score (fatigue/soreness/stress are inverted)
        score = (
            weights['hrv'] * self.hrv_score +
            weights['sleep_quality'] * self.sleep_quality +
            weights['sleep_hours'] * sleep_norm +
            weights['fatigue'] * (1 - self.fatigue_level) +
            weights['soreness'] * (1 - self.soreness_level) +
            weights['stress'] * (1 - self.stress_level)
        )

        return max(0.0, min(1.0, score))


def calculate_base_phase_frequency(
    weekly_volume: float,
    experience_level: ExperienceLevel = ExperienceLevel.BEGINNER,
    recovery: Optional[RecoveryMetrics] = None,
    max_session_duration: float = 90.0
) -> Tuple[int, Dict[str, Any]]:
    """
    Calculate running frequency for base building phase.

    Base phase prioritizes aerobic development with manageable frequency.
    All runs are Zone 2 (MAF), so recovery demands are moderate.

    Formula:
        base_freq = floor(Volume / 25) + 2
        Adjusted by recovery quality

    Constraints:
        - Minimum: 2 days (consistency)
        - Maximum: 4 days (recovery in base phase)
        - Each run should be >= 20 minutes

    Args:
        weekly_volume: Target weekly volume in minutes
        experience_level: Runner's experience level
        recovery: Recovery metrics (optional)
        max_session_duration: Maximum single session length

    Returns:
        Tuple of (frequency, breakdown_dict)
    """
    breakdown = {
        'weekly_volume': weekly_volume,
        'phase': 'base_building',
        'base_frequency': 0,
        'recovery_adjustment': 0,
        'experience_adjustment': 0,
        'duration_constraint_adjustment': 0,
    }

    # Base frequency calculation
    # At 50 min/week: 2 + 2 = 4 days (25 min avg - but minimum enforced)
    # At 100 min/week: 4 + 2 = 6 -> capped at 4
    base_freq = int(weekly_volume / 25) + 2
    breakdown['base_frequency'] = base_freq

    # Recovery adjustment
    if recovery is not None:
        recovery_score = recovery.recovery_score
        if recovery_score >= 0.8:
            breakdown['recovery_adjustment'] = 1
        elif recovery_score >= 0.6:
            breakdown['recovery_adjustment'] = 0
        else:
            breakdown['recovery_adjustment'] = -1

    # Experience adjustment (advanced can handle more frequency)
    if experience_level == ExperienceLevel.ADVANCED:
        breakdown['experience_adjustment'] = 1
    elif experience_level == ExperienceLevel.BEGINNER:
        breakdown['experience_adjustment'] = -1 if base_freq > 3 else 0

    # Duration constraint: ensure each run is meaningful (>= 20 min)
    min_run_duration = 20.0
    max_runs_by_duration = int(weekly_volume / min_run_duration)
    if base_freq > max_runs_by_duration:
        breakdown['duration_constraint_adjustment'] = max_runs_by_duration - base_freq

    # Calculate final frequency
    total_adjustment = (
        breakdown['recovery_adjustment'] +
        breakdown['experience_adjustment'] +
        breakdown['duration_constraint_adjustment']
    )

    frequency = base_freq + total_adjustment

    # Enforce base phase limits
    frequency = max(2, min(4, frequency))

    breakdown['final_frequency'] = frequency
    breakdown['avg_run_duration'] = weekly_volume / frequency if frequency > 0 else 0

    return frequency, breakdown


def calculate_eighty_twenty_frequency(
    weekly_volume: float,
    experience_level: ExperienceLevel = ExperienceLevel.INTERMEDIATE,
    recovery: Optional[RecoveryMetrics] = None,
    include_hard_sessions: int = 1
) -> Tuple[int, Dict[str, Any]]:
    """
    Calculate running frequency for 80/20 training phase.

    80/20 phase includes high-intensity sessions, requiring more
    careful frequency management to allow recovery between hard efforts.

    Formula:
        base_freq = floor(Volume / 40) + 3
        Adjusted by recovery and experience

    Constraints:
        - Minimum: 3 days (need variety for 80/20)
        - Maximum: 6 days (elite-level)
        - Hard sessions: 1-2 per week

    Args:
        weekly_volume: Target weekly volume in minutes
        experience_level: Runner's experience level
        recovery: Recovery metrics (optional)
        include_hard_sessions: Number of hard sessions (1 or 2)

    Returns:
        Tuple of (frequency, breakdown_dict)
    """
    breakdown = {
        'weekly_volume': weekly_volume,
        'phase': '80_20',
        'base_frequency': 0,
        'recovery_adjustment': 0,
        'experience_adjustment': 0,
        'hard_session_count': include_hard_sessions,
    }

    # Base frequency calculation
    # At 120 min/week: 3 + 3 = 6 -> may be capped
    # At 200 min/week: 5 + 3 = 8 -> capped at 6
    base_freq = int(weekly_volume / 40) + 3
    breakdown['base_frequency'] = base_freq

    # Recovery adjustment
    if recovery is not None:
        recovery_score = recovery.recovery_score
        if recovery_score >= 0.8:
            breakdown['recovery_adjustment'] = 1
        elif recovery_score >= 0.6:
            breakdown['recovery_adjustment'] = 0
        else:
            breakdown['recovery_adjustment'] = -1

    # Experience adjustment
    experience_adjustments = {
        ExperienceLevel.BEGINNER: -1,
        ExperienceLevel.NOVICE: 0,
        ExperienceLevel.INTERMEDIATE: 0,
        ExperienceLevel.ADVANCED: 1,
    }
    breakdown['experience_adjustment'] = experience_adjustments.get(
        experience_level, 0
    )

    # Calculate final frequency
    total_adjustment = (
        breakdown['recovery_adjustment'] +
        breakdown['experience_adjustment']
    )

    frequency = base_freq + total_adjustment

    # Enforce 80/20 phase limits based on experience
    if experience_level == ExperienceLevel.BEGINNER:
        frequency = max(3, min(4, frequency))
    elif experience_level == ExperienceLevel.NOVICE:
        frequency = max(3, min(5, frequency))
    else:
        frequency = max(3, min(6, frequency))

    breakdown['final_frequency'] = frequency
    breakdown['avg_run_duration'] = weekly_volume / frequency if frequency > 0 else 0
    breakdown['easy_session_count'] = frequency - include_hard_sessions

    return frequency, breakdown


def calculate_frequency(
    weekly_volume: float,
    phase: TrainingPhase,
    experience_level: ExperienceLevel = ExperienceLevel.INTERMEDIATE,
    recovery: Optional[RecoveryMetrics] = None,
    **kwargs
) -> Tuple[int, Dict[str, Any]]:
    """
    Master frequency calculation function.

    Routes to appropriate phase-specific calculation.

    Args:
        weekly_volume: Target weekly volume in minutes
        phase: Current training phase
        experience_level: Runner's experience level
        recovery: Recovery metrics (optional)
        **kwargs: Additional phase-specific arguments

    Returns:
        Tuple of (frequency, breakdown_dict)
    """
    if phase == TrainingPhase.BASE_BUILDING:
        return calculate_base_phase_frequency(
            weekly_volume,
            experience_level,
            recovery,
            kwargs.get('max_session_duration', 90.0)
        )

    elif phase == TrainingPhase.TRANSITION:
        # Transition: slightly higher frequency than base
        freq, breakdown = calculate_base_phase_frequency(
            weekly_volume,
            experience_level,
            recovery
        )
        # Allow one more day in transition
        freq = min(freq + 1, 5)
        breakdown['phase'] = 'transition'
        breakdown['final_frequency'] = freq
        return freq, breakdown

    elif phase == TrainingPhase.EIGHTY_TWENTY:
        return calculate_eighty_twenty_frequency(
            weekly_volume,
            experience_level,
            recovery,
            kwargs.get('include_hard_sessions', 1)
        )

    elif phase == TrainingPhase.RACE_SPECIFIC:
        # Race specific: similar to 80/20 but may have 2 hard sessions
        return calculate_eighty_twenty_frequency(
            weekly_volume,
            experience_level,
            recovery,
            include_hard_sessions=2
        )

    elif phase == TrainingPhase.TAPER:
        # Taper: reduce frequency by 1-2 from current
        freq, breakdown = calculate_eighty_twenty_frequency(
            weekly_volume,
            experience_level,
            recovery
        )
        freq = max(2, freq - 1)
        breakdown['phase'] = 'taper'
        breakdown['final_frequency'] = freq
        return freq, breakdown

    elif phase == TrainingPhase.RECOVERY:
        # Recovery: minimum frequency, easy only
        freq = max(2, min(3, int(weekly_volume / 30)))
        return freq, {
            'phase': 'recovery',
            'weekly_volume': weekly_volume,
            'final_frequency': freq,
            'avg_run_duration': weekly_volume / freq if freq > 0 else 0,
        }

    else:
        # Default to base phase
        return calculate_base_phase_frequency(
            weekly_volume,
            experience_level,
            recovery
        )


def validate_frequency_duration(
    frequency: int,
    weekly_volume: float,
    min_run_duration: float = 20.0,
    max_run_duration: float = 120.0
) -> Tuple[bool, str]:
    """
    Validate that frequency produces sensible run durations.

    Args:
        frequency: Proposed running days per week
        weekly_volume: Total weekly volume in minutes
        min_run_duration: Minimum acceptable run duration
        max_run_duration: Maximum acceptable run duration

    Returns:
        Tuple of (is_valid, message)
    """
    if frequency <= 0:
        return False, "Frequency must be positive"

    avg_duration = weekly_volume / frequency

    if avg_duration < min_run_duration:
        return False, f"Average run duration ({avg_duration:.0f} min) below minimum ({min_run_duration:.0f} min)"

    if avg_duration > max_run_duration:
        return False, f"Average run duration ({avg_duration:.0f} min) exceeds maximum ({max_run_duration:.0f} min)"

    return True, "Valid"


def adjust_frequency_for_volume(
    target_frequency: int,
    weekly_volume: float,
    min_run_duration: float = 20.0,
    max_run_duration: float = 90.0
) -> int:
    """
    Adjust frequency to ensure run durations are within bounds.

    Args:
        target_frequency: Desired frequency
        weekly_volume: Total weekly volume in minutes
        min_run_duration: Minimum acceptable run duration
        max_run_duration: Maximum acceptable run duration

    Returns:
        Adjusted frequency
    """
    # If runs too short, reduce frequency
    while target_frequency > 1:
        avg_duration = weekly_volume / target_frequency
        if avg_duration >= min_run_duration:
            break
        target_frequency -= 1

    # If runs too long, increase frequency
    while target_frequency < 7:
        avg_duration = weekly_volume / target_frequency
        if avg_duration <= max_run_duration:
            break
        target_frequency += 1

    return target_frequency


def get_frequency_recommendations(
    weekly_volume: float,
    experience_level: ExperienceLevel,
    phase: TrainingPhase
) -> Dict[str, Any]:
    """
    Get frequency recommendations with context.

    Args:
        weekly_volume: Target weekly volume
        experience_level: Runner's experience
        phase: Training phase

    Returns:
        Dictionary with recommendations and explanations
    """
    frequency, breakdown = calculate_frequency(
        weekly_volume,
        phase,
        experience_level
    )

    avg_duration = weekly_volume / frequency

    recommendations = {
        'frequency': frequency,
        'avg_duration': avg_duration,
        'phase': phase.value,
        'experience': experience_level.value,
        'weekly_volume': weekly_volume,
        'breakdown': breakdown,
    }

    # Add contextual advice
    if avg_duration < 25:
        recommendations['advice'] = (
            "Consider reducing frequency to allow longer runs. "
            "Runs under 25 minutes provide limited aerobic benefit."
        )
    elif avg_duration > 75 and phase == TrainingPhase.BASE_BUILDING:
        recommendations['advice'] = (
            "Consider adding another day to distribute volume. "
            "Very long easy runs increase injury risk without proportional benefit."
        )
    elif frequency >= 5 and experience_level in [ExperienceLevel.BEGINNER, ExperienceLevel.NOVICE]:
        recommendations['advice'] = (
            "High frequency for your experience level. "
            "Ensure adequate rest between runs and monitor for fatigue."
        )
    else:
        recommendations['advice'] = "Frequency looks appropriate for your volume and experience."

    return recommendations


if __name__ == '__main__':
    print("Testing Frequency Equations...")
    print("=" * 60)

    # Test cases
    test_cases = [
        (60, ExperienceLevel.BEGINNER, TrainingPhase.BASE_BUILDING),
        (120, ExperienceLevel.INTERMEDIATE, TrainingPhase.BASE_BUILDING),
        (150, ExperienceLevel.INTERMEDIATE, TrainingPhase.EIGHTY_TWENTY),
        (200, ExperienceLevel.ADVANCED, TrainingPhase.EIGHTY_TWENTY),
        (180, ExperienceLevel.ADVANCED, TrainingPhase.RACE_SPECIFIC),
        (100, ExperienceLevel.INTERMEDIATE, TrainingPhase.TAPER),
    ]

    for volume, exp, phase in test_cases:
        freq, breakdown = calculate_frequency(volume, phase, exp)
        avg_dur = volume / freq
        print(f"\nVolume: {volume} min | {exp.value} | {phase.value}")
        print(f"  Frequency: {freq} days/week")
        print(f"  Avg Duration: {avg_dur:.0f} min/run")

    # Test with recovery metrics
    print("\n" + "=" * 60)
    print("Testing with recovery metrics...")

    good_recovery = RecoveryMetrics(
        hrv_score=0.85,
        sleep_quality=0.9,
        sleep_hours=8.0,
        fatigue_level=0.2,
        soreness_level=0.1,
        stress_level=0.2
    )

    poor_recovery = RecoveryMetrics(
        hrv_score=0.5,
        sleep_quality=0.4,
        sleep_hours=5.5,
        fatigue_level=0.7,
        soreness_level=0.6,
        stress_level=0.8
    )

    print(f"\nGood recovery score: {good_recovery.recovery_score:.2f}")
    freq_good, _ = calculate_base_phase_frequency(
        120, ExperienceLevel.INTERMEDIATE, good_recovery
    )
    print(f"  Frequency with good recovery: {freq_good}")

    print(f"\nPoor recovery score: {poor_recovery.recovery_score:.2f}")
    freq_poor, _ = calculate_base_phase_frequency(
        120, ExperienceLevel.INTERMEDIATE, poor_recovery
    )
    print(f"  Frequency with poor recovery: {freq_poor}")

    print("\n" + "=" * 60)
    print("All tests completed!")
