"""
MAF Heart Rate and Initial Volume Calculations.

Based on:
- Maffetone, P. (2010). The Big Book of Endurance Training and Racing
- Refined with fitness-level scaling and HealthKit integration

These equations establish baseline physiological thresholds and starting conditions
for a running plan.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, Tuple
import math


class ExperienceLevel(Enum):
    """Runner experience classification."""
    BEGINNER = "beginner"           # < 6 months consistent running
    NOVICE = "novice"               # 6 months - 2 years
    INTERMEDIATE = "intermediate"   # 2-4 years consistent
    ADVANCED = "advanced"           # 4+ years, no significant injuries


class HealthStatus(Enum):
    """Health condition classification."""
    OPTIMAL = "optimal"             # No health issues
    MINOR_ISSUES = "minor_issues"   # Minor issues, generally healthy
    RECOVERING = "recovering"       # Recovering from illness/injury
    CHRONIC = "chronic"             # Chronic condition affecting training


@dataclass
class AthleteProfile:
    """
    Core athlete profile for personalized calculations.

    This profile drives all equation adjustments for individual runners.
    """
    # Demographics
    age: int
    gender: str  # 'male' or 'female'
    weight_kg: float
    height_cm: float

    # Experience
    experience_level: ExperienceLevel = ExperienceLevel.BEGINNER
    consistent_training_years: float = 0.0
    injury_free_years: float = 0.0

    # Health
    health_status: HealthStatus = HealthStatus.OPTIMAL
    on_hr_medication: bool = False

    # Measured or estimated physiology
    hr_rest: Optional[float] = None      # From HealthKit
    hr_max: Optional[float] = None       # From HealthKit or field test
    vo2max_estimate: Optional[float] = None  # From HealthKit

    # Recent activity (from HealthKit)
    recent_4wk_avg_volume: Optional[float] = None  # min/week
    recent_easy_run_hr: Optional[float] = None     # avg HR at easy pace

    # Availability
    available_days_per_week: int = 5
    max_session_duration: float = 90.0   # minutes

    @property
    def bmi(self) -> float:
        """Calculate Body Mass Index."""
        height_m = self.height_cm / 100
        return self.weight_kg / (height_m ** 2)

    @property
    def is_obese(self) -> bool:
        """Check if BMI indicates obesity (>30)."""
        return self.bmi > 30

    @property
    def is_overweight(self) -> bool:
        """Check if BMI indicates overweight (>25)."""
        return self.bmi > 25


def calculate_maf_hr(
    age: int,
    experience_level: ExperienceLevel = ExperienceLevel.BEGINNER,
    consistent_training_years: float = 0.0,
    injury_free_years: float = 0.0,
    health_status: HealthStatus = HealthStatus.OPTIMAL,
    bmi: Optional[float] = None,
    on_hr_medication: bool = False,
    observed_easy_hr: Optional[float] = None
) -> Tuple[int, Dict[str, Any]]:
    """
    Calculate MAF (Maximum Aerobic Function) Heart Rate.

    The MAF HR defines the upper limit of Zone 2 aerobic training,
    where fat oxidation is maximized and aerobic base is built.

    Base Formula: MAF_HR = 180 - Age + Adjustments

    Args:
        age: Age in years
        experience_level: Runner's experience classification
        consistent_training_years: Years of consistent training
        injury_free_years: Years without significant injury
        health_status: Current health status
        bmi: Body Mass Index (optional)
        on_hr_medication: Whether on HR-affecting medication
        observed_easy_hr: Average HR from easy runs (from HealthKit)

    Returns:
        Tuple of (maf_hr, breakdown_dict)
    """
    # Base MAF formula
    base_maf = 180 - age

    adjustments = {
        'base': base_maf,
        'experience': 0,
        'health': 0,
        'fitness': 0,
        'medication': 0
    }

    # Experience adjustment (+5 to +10)
    if experience_level == ExperienceLevel.ADVANCED:
        if injury_free_years >= 2 and consistent_training_years >= 4:
            adjustments['experience'] = 10
        else:
            adjustments['experience'] = 5
    elif experience_level == ExperienceLevel.INTERMEDIATE:
        if consistent_training_years >= 2:
            adjustments['experience'] = 5
    # Beginner/Novice: no adjustment

    # Health adjustment (-5 to -10)
    if health_status == HealthStatus.CHRONIC:
        adjustments['health'] = -10
    elif health_status == HealthStatus.RECOVERING:
        adjustments['health'] = -10
    elif health_status == HealthStatus.MINOR_ISSUES:
        adjustments['health'] = -5

    # BMI adjustment (additional -5 for obesity)
    if bmi is not None and bmi > 30:
        adjustments['health'] = min(adjustments['health'], -5)
        adjustments['health'] -= 5  # Additional penalty

    # Medication adjustment
    if on_hr_medication:
        adjustments['medication'] = -10

    # Fitness adjustment (based on observed vs predicted easy HR)
    if observed_easy_hr is not None:
        predicted_easy_hr = 0.65 * (220 - age)  # 65% of estimated max
        hr_diff = observed_easy_hr - predicted_easy_hr
        # If actual easy HR is lower than predicted, runner is more fit
        adjustments['fitness'] = int(max(-5, min(5, -hr_diff / 2)))

    # Calculate final MAF HR
    total_adjustment = (
        adjustments['experience'] +
        adjustments['health'] +
        adjustments['fitness'] +
        adjustments['medication']
    )

    maf_hr = base_maf + total_adjustment

    # Enforce physiological bounds
    maf_hr = max(100, min(170, maf_hr))

    adjustments['total_adjustment'] = total_adjustment
    adjustments['final_maf_hr'] = maf_hr

    return int(maf_hr), adjustments


def calculate_initial_volume(
    experience_level: ExperienceLevel = ExperienceLevel.BEGINNER,
    recent_4wk_avg_volume: Optional[float] = None,
    vo2max_estimate: Optional[float] = None,
    hr_rest: Optional[float] = None,
    bmi: Optional[float] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate recommended initial weekly running volume.

    This provides a safe starting point that respects the athlete's
    current fitness while avoiding excessive initial load.

    Formula:
        Initial_Volume = base_volume × history_multiplier × fitness_multiplier × bmi_multiplier

    Args:
        experience_level: Runner's experience classification
        recent_4wk_avg_volume: Recent average weekly volume (min)
        vo2max_estimate: Estimated VO2max from HealthKit
        hr_rest: Resting heart rate
        bmi: Body Mass Index

    Returns:
        Tuple of (initial_volume_minutes, breakdown_dict)
    """
    # Base volume (moderate starting point)
    base_volume = 90.0  # minutes per week

    breakdown = {
        'base_volume': base_volume,
        'history_multiplier': 1.0,
        'fitness_multiplier': 1.0,
        'bmi_multiplier': 1.0,
        'experience_cap': None
    }

    # History multiplier: respect recent training
    if recent_4wk_avg_volume is not None and recent_4wk_avg_volume > 0:
        # Scale based on recent activity, capped at 2x base
        history_mult = recent_4wk_avg_volume / base_volume
        breakdown['history_multiplier'] = max(0.5, min(2.0, history_mult))
    else:
        # No history: conservative start
        breakdown['history_multiplier'] = 0.67

    # Fitness multiplier: adjust for cardiovascular fitness
    if vo2max_estimate is not None:
        if vo2max_estimate > 50:
            breakdown['fitness_multiplier'] = 1.3
        elif vo2max_estimate > 40:
            breakdown['fitness_multiplier'] = 1.1
        elif vo2max_estimate > 35:
            breakdown['fitness_multiplier'] = 1.0
        else:
            breakdown['fitness_multiplier'] = 0.8
    elif hr_rest is not None:
        # Use resting HR as fitness proxy
        if hr_rest < 55:
            breakdown['fitness_multiplier'] = 1.3
        elif hr_rest < 65:
            breakdown['fitness_multiplier'] = 1.1
        elif hr_rest < 75:
            breakdown['fitness_multiplier'] = 1.0
        else:
            breakdown['fitness_multiplier'] = 0.8

    # BMI multiplier: gradual reduction for higher BMI
    if bmi is not None:
        if bmi <= 25:
            breakdown['bmi_multiplier'] = 1.0
        elif bmi <= 35:
            # 2% reduction per BMI point above 25, capping at 0.8
            reduction = 0.02 * (bmi - 25)
            breakdown['bmi_multiplier'] = max(0.8, 1.0 - reduction)
        else:
            breakdown['bmi_multiplier'] = 0.8

    # Calculate initial volume
    initial_volume = (
        base_volume *
        breakdown['history_multiplier'] *
        breakdown['fitness_multiplier'] *
        breakdown['bmi_multiplier']
    )

    # Apply experience-based caps
    experience_caps = {
        ExperienceLevel.BEGINNER: (30, 120),
        ExperienceLevel.NOVICE: (45, 180),
        ExperienceLevel.INTERMEDIATE: (60, 240),
        ExperienceLevel.ADVANCED: (60, 300),
    }

    min_vol, max_vol = experience_caps.get(
        experience_level,
        (30, 120)
    )

    breakdown['experience_cap'] = (min_vol, max_vol)
    initial_volume = max(min_vol, min(max_vol, initial_volume))

    breakdown['calculated_volume'] = initial_volume

    return round(initial_volume, 0), breakdown


def calculate_hr_zones(
    maf_hr: int,
    hr_max: Optional[int] = None,
    age: Optional[int] = None
) -> Dict[str, Tuple[int, int]]:
    """
    Calculate heart rate training zones based on MAF HR.

    Zone system:
        Zone 1: Recovery (very easy)
        Zone 2: Aerobic base (MAF training)
        Zone 3: Tempo (threshold)
        Zone 4: VO2max intervals
        Zone 5: Anaerobic (sprints)

    Args:
        maf_hr: MAF heart rate (Zone 2 ceiling)
        hr_max: Maximum heart rate (measured or estimated)
        age: Age for hr_max estimation if not provided

    Returns:
        Dictionary of zone name -> (lower_bound, upper_bound)
    """
    if hr_max is None:
        if age is None:
            raise ValueError("Must provide either hr_max or age")
        hr_max = 220 - age

    # MAF HR is the ceiling of Zone 2
    # Zone 2 spans about 10 bpm below MAF

    zones = {
        'zone_1_recovery': (maf_hr - 20, maf_hr - 10),
        'zone_2_aerobic': (maf_hr - 10, maf_hr),
        'zone_3_tempo': (maf_hr + 1, int(hr_max * 0.88)),
        'zone_4_threshold': (int(hr_max * 0.88) + 1, int(hr_max * 0.95)),
        'zone_5_anaerobic': (int(hr_max * 0.95) + 1, hr_max),
    }

    return zones


def calculate_bmi_stress_modifier(bmi: float) -> float:
    """
    Calculate stress modifier based on BMI.

    Higher BMI individuals experience greater cardiovascular stress
    at the same relative intensity. This modifier inflates TRIMP
    calculations to reflect true physiological load.

    Formula:
        modifier = 1.0 if BMI <= 25
        modifier = 1.0 + 0.03 × (BMI - 25) if 25 < BMI <= 30
        modifier = 1.15 + 0.05 × (BMI - 30) if 30 < BMI <= 40
        modifier = 1.65 if BMI > 40

    Args:
        bmi: Body Mass Index

    Returns:
        Stress modifier (1.0 = no adjustment)
    """
    if bmi <= 25:
        return 1.0
    elif bmi <= 30:
        return 1.0 + 0.03 * (bmi - 25)  # Up to 1.15
    elif bmi <= 40:
        return 1.15 + 0.05 * (bmi - 30)  # Up to 1.65
    else:
        return 1.65


def calculate_max_delta_v_cap(
    experience_level: ExperienceLevel,
    bmi: Optional[float] = None
) -> float:
    """
    Calculate maximum positive Delta V cap based on experience and BMI.

    Beginners and overweight/obese runners should have more conservative
    volume increases to account for unperceived physiological stress.

    Args:
        experience_level: Runner's experience classification
        bmi: Body Mass Index (optional)

    Returns:
        Maximum positive Delta V as decimal (e.g., 0.10 = 10%)
    """
    # Base caps by experience
    experience_caps = {
        ExperienceLevel.BEGINNER: 0.10,
        ExperienceLevel.NOVICE: 0.12,
        ExperienceLevel.INTERMEDIATE: 0.15,
        ExperienceLevel.ADVANCED: 0.20,
    }

    cap = experience_caps.get(experience_level, 0.10)

    # Additional reduction for high BMI
    if bmi is not None:
        if bmi > 35:
            cap = min(cap, 0.08)
        elif bmi > 30:
            cap = min(cap, 0.10)
        elif bmi > 28:
            cap = min(cap, 0.12)

    return cap


def get_athlete_adjustments(profile: AthleteProfile) -> Dict[str, Any]:
    """
    Calculate all personalized adjustments for an athlete profile.

    This is a convenience function that computes MAF HR, initial volume,
    BMI modifier, and Delta V cap in one call.

    Args:
        profile: Complete athlete profile

    Returns:
        Dictionary with all calculated adjustments
    """
    # Calculate MAF HR
    maf_hr, maf_breakdown = calculate_maf_hr(
        age=profile.age,
        experience_level=profile.experience_level,
        consistent_training_years=profile.consistent_training_years,
        injury_free_years=profile.injury_free_years,
        health_status=profile.health_status,
        bmi=profile.bmi,
        on_hr_medication=profile.on_hr_medication,
        observed_easy_hr=profile.recent_easy_run_hr
    )

    # Calculate initial volume
    initial_volume, volume_breakdown = calculate_initial_volume(
        experience_level=profile.experience_level,
        recent_4wk_avg_volume=profile.recent_4wk_avg_volume,
        vo2max_estimate=profile.vo2max_estimate,
        hr_rest=profile.hr_rest,
        bmi=profile.bmi
    )

    # Calculate HR zones
    hr_zones = calculate_hr_zones(
        maf_hr=maf_hr,
        hr_max=profile.hr_max,
        age=profile.age
    )

    # Calculate stress modifier
    bmi_stress_modifier = calculate_bmi_stress_modifier(profile.bmi)

    # Calculate Delta V cap
    max_delta_v = calculate_max_delta_v_cap(
        experience_level=profile.experience_level,
        bmi=profile.bmi
    )

    return {
        'maf_hr': maf_hr,
        'maf_breakdown': maf_breakdown,
        'initial_volume': initial_volume,
        'volume_breakdown': volume_breakdown,
        'hr_zones': hr_zones,
        'bmi': profile.bmi,
        'bmi_stress_modifier': bmi_stress_modifier,
        'max_delta_v': max_delta_v,
        'experience_level': profile.experience_level.value,
    }


if __name__ == '__main__':
    print("Testing MAF and Initial Volume calculations...")
    print("=" * 60)

    # Test case 1: Beginner, overweight
    profile1 = AthleteProfile(
        age=35,
        gender='male',
        weight_kg=95,
        height_cm=175,
        experience_level=ExperienceLevel.BEGINNER,
        health_status=HealthStatus.OPTIMAL,
        hr_rest=72,
    )

    adjustments1 = get_athlete_adjustments(profile1)
    print(f"\nCase 1: Beginner, BMI {profile1.bmi:.1f}")
    print(f"  MAF HR: {adjustments1['maf_hr']} bpm")
    print(f"  Initial Volume: {adjustments1['initial_volume']} min/week")
    print(f"  BMI Stress Modifier: {adjustments1['bmi_stress_modifier']:.2f}x")
    print(f"  Max Delta V: +{adjustments1['max_delta_v']*100:.0f}%")

    # Test case 2: Experienced, fit
    profile2 = AthleteProfile(
        age=40,
        gender='female',
        weight_kg=58,
        height_cm=165,
        experience_level=ExperienceLevel.ADVANCED,
        consistent_training_years=6,
        injury_free_years=3,
        health_status=HealthStatus.OPTIMAL,
        hr_rest=52,
        vo2max_estimate=48,
        recent_4wk_avg_volume=180,
    )

    adjustments2 = get_athlete_adjustments(profile2)
    print(f"\nCase 2: Advanced, BMI {profile2.bmi:.1f}")
    print(f"  MAF HR: {adjustments2['maf_hr']} bpm")
    print(f"  Initial Volume: {adjustments2['initial_volume']} min/week")
    print(f"  BMI Stress Modifier: {adjustments2['bmi_stress_modifier']:.2f}x")
    print(f"  Max Delta V: +{adjustments2['max_delta_v']*100:.0f}%")
    print(f"  HR Zones: {adjustments2['hr_zones']}")

    # Test case 3: Obese beginner
    profile3 = AthleteProfile(
        age=45,
        gender='male',
        weight_kg=120,
        height_cm=180,
        experience_level=ExperienceLevel.BEGINNER,
        health_status=HealthStatus.MINOR_ISSUES,
        hr_rest=78,
    )

    adjustments3 = get_athlete_adjustments(profile3)
    print(f"\nCase 3: Obese beginner, BMI {profile3.bmi:.1f}")
    print(f"  MAF HR: {adjustments3['maf_hr']} bpm")
    print(f"  Initial Volume: {adjustments3['initial_volume']} min/week")
    print(f"  BMI Stress Modifier: {adjustments3['bmi_stress_modifier']:.2f}x")
    print(f"  Max Delta V: +{adjustments3['max_delta_v']*100:.0f}%")

    print("\n" + "=" * 60)
    print("All tests completed!")
