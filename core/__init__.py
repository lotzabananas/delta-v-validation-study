"""
Core metrics and equation implementations for running plan management.

This package provides a complete set of equations for:
- Load quantification (TRIMP, EWMA, ACWR)
- Volume adjustment (Delta V)
- Physiological thresholds (MAF HR, zones)
- Training frequency
- Workout distribution
- Schedule optimization

See RUN_PLAN_EQUATIONS.md for comprehensive documentation.
"""

# Load quantification
from .metrics import (
    calculate_trimp,
    calculate_ewma,
    calculate_acwr,
    calculate_acwr_series,
    classify_acwr_zone,
)

# Delta V equation
from .delta_v import (
    DeltaVParams,
    calculate_delta_v,
    apply_delta_v,
    get_delta_v_summary,
)

# MAF and physiological thresholds
from .maf import (
    AthleteProfile,
    ExperienceLevel,
    HealthStatus,
    calculate_maf_hr,
    calculate_initial_volume,
    calculate_bmi_stress_modifier,
    calculate_max_delta_v_cap,
    calculate_hr_zones,
    get_athlete_adjustments,
)

# Frequency calculations
from .frequency import (
    TrainingPhase,
    RecoveryMetrics,
    calculate_frequency,
    calculate_base_phase_frequency,
    calculate_eighty_twenty_frequency,
)

# Workout splits
from .workout_splits import (
    WorkoutType,
    IntensityZone,
    WorkoutSession,
    WeeklyWorkoutPlan,
    calculate_splits,
    calculate_base_phase_splits,
    calculate_eighty_twenty_splits,
)

# Scheduling
from .scheduling import (
    DayOfWeek,
    DayPreference,
    WeekSchedule,
    ScheduledWorkout,
    assign_days,
    calculate_max_consecutive_runs,
    format_weekly_schedule,
)

# Unified engine
from .run_plan_engine import (
    RunPlanEngine,
    RunPlanState,
    PlanStatus,
    TrainingWeekLog,
)

__all__ = [
    # Metrics
    'calculate_trimp',
    'calculate_ewma',
    'calculate_acwr',
    'calculate_acwr_series',
    'classify_acwr_zone',
    # Delta V
    'DeltaVParams',
    'calculate_delta_v',
    'apply_delta_v',
    'get_delta_v_summary',
    # MAF
    'AthleteProfile',
    'ExperienceLevel',
    'HealthStatus',
    'calculate_maf_hr',
    'calculate_initial_volume',
    'calculate_bmi_stress_modifier',
    'calculate_max_delta_v_cap',
    'calculate_hr_zones',
    'get_athlete_adjustments',
    # Frequency
    'TrainingPhase',
    'RecoveryMetrics',
    'calculate_frequency',
    'calculate_base_phase_frequency',
    'calculate_eighty_twenty_frequency',
    # Splits
    'WorkoutType',
    'IntensityZone',
    'WorkoutSession',
    'WeeklyWorkoutPlan',
    'calculate_splits',
    'calculate_base_phase_splits',
    'calculate_eighty_twenty_splits',
    # Scheduling
    'DayOfWeek',
    'DayPreference',
    'WeekSchedule',
    'ScheduledWorkout',
    'assign_days',
    'calculate_max_consecutive_runs',
    'format_weekly_schedule',
    # Engine
    'RunPlanEngine',
    'RunPlanState',
    'PlanStatus',
    'TrainingWeekLog',
]
