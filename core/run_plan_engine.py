"""
Run Plan Engine: Unified system integrating all equations.

This module ties together all the individual equation modules to provide
a complete running plan management system, including:
- Initial plan generation
- Weekly updates based on ACWR
- Phase transitions
- Deload scheduling

The engine operates as a state machine, tracking athlete progress and
adjusting the plan dynamically based on training load and recovery.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple, Set
import numpy as np

# Import all equation modules
from .metrics import (
    calculate_trimp,
    calculate_acwr,
    calculate_ewma,
    classify_acwr_zone,
)
from .delta_v import (
    DeltaVParams,
    calculate_delta_v,
    apply_delta_v,
    get_delta_v_summary,
)
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
from .frequency import (
    TrainingPhase,
    RecoveryMetrics,
    calculate_frequency,
    validate_frequency_duration,
)
from .workout_splits import (
    WorkoutSession,
    WorkoutType,
    WeeklyWorkoutPlan,
    calculate_splits,
)
from .scheduling import (
    DayOfWeek,
    DayPreference,
    WeekSchedule,
    assign_days,
    format_weekly_schedule,
)


class PlanStatus(Enum):
    """Current status of the training plan."""
    ACTIVE = "active"               # Normal training
    DELOAD = "deload"               # Scheduled deload week
    RECOVERY = "recovery"           # Injury/illness recovery
    TRANSITION = "transition"       # Phase transition
    PAUSED = "paused"               # Temporarily paused


@dataclass
class TrainingWeekLog:
    """
    Log of a completed training week.

    Used for tracking history and computing ACWR.
    """
    week_number: int
    week_start: date
    planned_volume: float
    actual_volume: float
    planned_frequency: int
    actual_frequency: int
    phase: TrainingPhase
    acwr: float
    delta_v_applied: float
    trimp_daily: List[float] = field(default_factory=list)
    notes: str = ""

    @property
    def adherence_rate(self) -> float:
        """Calculate adherence to planned volume."""
        if self.planned_volume == 0:
            return 1.0
        return min(1.5, self.actual_volume / self.planned_volume)


@dataclass
class RunPlanState:
    """
    Complete state of a running plan.

    This is the main data structure tracking all plan parameters.
    """
    # Athlete info
    athlete: AthleteProfile

    # Current plan parameters
    current_volume: float
    current_frequency: int
    current_phase: TrainingPhase
    status: PlanStatus = PlanStatus.ACTIVE

    # Computed values
    maf_hr: int = 0
    hr_zones: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    max_delta_v: float = 0.15
    bmi_stress_modifier: float = 1.0

    # ACWR tracking
    trimp_history: List[float] = field(default_factory=list)
    current_acwr: float = 1.0
    acwr_zone: str = "optimal"
    consecutive_high_acwr_weeks: int = 0

    # Week tracking
    week_number: int = 1
    weeks_in_current_phase: int = 0
    weeks_since_deload: int = 0
    week_logs: List[TrainingWeekLog] = field(default_factory=list)

    # Scheduling
    available_days: Set[DayOfWeek] = field(default_factory=lambda: set(DayOfWeek))
    day_preferences: List[DayPreference] = field(default_factory=list)

    # Current week's plan
    current_workout_plan: Optional[WeeklyWorkoutPlan] = None
    current_schedule: Optional[WeekSchedule] = None


class RunPlanEngine:
    """
    Main engine for managing running plans.

    This class orchestrates all equation modules to provide
    a unified interface for plan management.
    """

    def __init__(
        self,
        delta_v_params: Optional[DeltaVParams] = None,
        use_optimized_params: bool = True
    ):
        """
        Initialize the run plan engine.

        Args:
            delta_v_params: Custom Delta V parameters (optional)
            use_optimized_params: Whether to use optimized parameters
        """
        if delta_v_params is not None:
            self.delta_v_params = delta_v_params
        elif use_optimized_params:
            # Use validated optimized parameters
            self.delta_v_params = DeltaVParams(
                threshold_low=0.8,
                threshold_optimal_high=1.3,
                threshold_caution=1.5,
                threshold_critical=2.0,
                green_base=0.30,
                green_min=0.08,
                green_max=0.20,
                low_base=0.31,
                low_min=0.14,
                low_max=0.17,
                caution_value=0.02,
                red_base=-0.26,
                red_min=-0.16,
                red_max=-0.05,
                critical_value=-0.27,
            )
        else:
            self.delta_v_params = DeltaVParams()

    def initialize_plan(
        self,
        athlete: AthleteProfile,
        start_date: Optional[date] = None,
        available_days: Optional[Set[DayOfWeek]] = None,
        preferences: Optional[List[DayPreference]] = None
    ) -> RunPlanState:
        """
        Initialize a new running plan for an athlete.

        This performs all initial calculations and generates the first week's plan.

        Args:
            athlete: Complete athlete profile
            start_date: Plan start date (default: today)
            available_days: Days athlete can run
            preferences: Day preferences

        Returns:
            Initialized RunPlanState
        """
        # Get all athlete adjustments
        adjustments = get_athlete_adjustments(athlete)

        # Create state
        state = RunPlanState(
            athlete=athlete,
            current_volume=adjustments['initial_volume'],
            current_frequency=0,  # Will be calculated
            current_phase=TrainingPhase.BASE_BUILDING,
            maf_hr=adjustments['maf_hr'],
            hr_zones=adjustments['hr_zones'],
            max_delta_v=adjustments['max_delta_v'],
            bmi_stress_modifier=adjustments['bmi_stress_modifier'],
            current_acwr=1.0,  # Start at neutral
            acwr_zone="optimal",
            available_days=available_days or set(DayOfWeek),
            day_preferences=preferences or [],
        )

        # Calculate initial frequency
        state.current_frequency, _ = calculate_frequency(
            state.current_volume,
            state.current_phase,
            athlete.experience_level
        )

        # Generate first week's plan
        state = self._generate_weekly_plan(state)

        return state

    def _generate_weekly_plan(self, state: RunPlanState) -> RunPlanState:
        """
        Generate workout plan and schedule for the current week.

        Args:
            state: Current plan state

        Returns:
            Updated state with new plan
        """
        # Determine hard session count based on phase and frequency
        if state.current_phase == TrainingPhase.BASE_BUILDING:
            hard_sessions = 0
        elif state.current_phase == TrainingPhase.TRANSITION:
            hard_sessions = 1
        elif state.current_phase in [TrainingPhase.EIGHTY_TWENTY, TrainingPhase.RACE_SPECIFIC]:
            hard_sessions = 2 if state.current_frequency >= 5 else 1
        else:
            hard_sessions = 0

        # Generate workout splits
        workout_plan = calculate_splits(
            weekly_volume=state.current_volume,
            frequency=state.current_frequency,
            phase=state.current_phase,
            experience_level=state.athlete.experience_level,
            hard_session_count=hard_sessions
        )

        state.current_workout_plan = workout_plan

        # Assign days
        schedule = assign_days(
            workout_plan=workout_plan,
            experience_level=state.athlete.experience_level,
            available_days=state.available_days,
            preferences=state.day_preferences,
            bmi=state.athlete.bmi,
            weekly_volume=state.current_volume
        )

        state.current_schedule = schedule

        # Set HR targets on sessions
        self._apply_hr_targets(state)

        return state

    def _apply_hr_targets(self, state: RunPlanState):
        """Apply HR zone targets to workout sessions."""
        if state.current_workout_plan is None:
            return

        for session in state.current_workout_plan.sessions:
            zone = session.intensity_zone.value
            if zone in state.hr_zones:
                session.hr_target_low, session.hr_target_high = state.hr_zones[zone]

    def update_with_training_data(
        self,
        state: RunPlanState,
        daily_trimp: List[float],
        actual_volume: float,
        actual_frequency: int,
        recovery: Optional[RecoveryMetrics] = None
    ) -> RunPlanState:
        """
        Update plan based on completed training week.

        This is the main weekly update function that:
        1. Records completed training
        2. Calculates new ACWR
        3. Computes Delta V
        4. Generates next week's plan

        Args:
            state: Current plan state
            daily_trimp: TRIMP values for each day of the week
            actual_volume: Actual volume completed (minutes)
            actual_frequency: Actual number of runs
            recovery: Recovery metrics (optional)

        Returns:
            Updated RunPlanState for next week
        """
        # Add daily TRIMP to history
        state.trimp_history.extend(daily_trimp)

        # Apply BMI stress modifier to TRIMP
        if state.bmi_stress_modifier > 1.0:
            adjusted_trimp = [t * state.bmi_stress_modifier for t in daily_trimp]
            trimp_for_acwr = state.trimp_history[:-len(daily_trimp)] + adjusted_trimp
        else:
            trimp_for_acwr = state.trimp_history

        # Calculate ACWR (need at least 14 days)
        if len(trimp_for_acwr) >= 14:
            state.current_acwr, _, _ = calculate_acwr(np.array(trimp_for_acwr))
        else:
            state.current_acwr = 1.0  # Default to neutral

        state.acwr_zone = classify_acwr_zone(state.current_acwr)

        # Track consecutive high ACWR weeks
        if state.current_acwr >= 1.5:
            state.consecutive_high_acwr_weeks += 1
        else:
            state.consecutive_high_acwr_weeks = 0

        # Log the completed week
        log = TrainingWeekLog(
            week_number=state.week_number,
            week_start=date.today() - timedelta(days=7),
            planned_volume=state.current_volume,
            actual_volume=actual_volume,
            planned_frequency=state.current_frequency,
            actual_frequency=actual_frequency,
            phase=state.current_phase,
            acwr=state.current_acwr,
            delta_v_applied=0,  # Will be set below
            trimp_daily=daily_trimp,
        )

        # Calculate Delta V
        delta_v, zone, flag = calculate_delta_v(
            state.current_acwr,
            self.delta_v_params,
            state.consecutive_high_acwr_weeks
        )

        # Apply experience/BMI cap
        if delta_v > 0:
            delta_v = min(delta_v, state.max_delta_v)

        log.delta_v_applied = delta_v
        state.week_logs.append(log)

        # Check for deload
        state.weeks_since_deload += 1
        needs_deload = (
            state.weeks_since_deload >= 4 or
            flag or
            state.consecutive_high_acwr_weeks >= 2
        )

        if needs_deload:
            state = self._apply_deload(state)
        else:
            # Apply Delta V to volume
            state.current_volume = apply_delta_v(state.current_volume, delta_v)

        # Check phase transition
        state = self._check_phase_transition(state)

        # Recalculate frequency
        state.current_frequency, _ = calculate_frequency(
            state.current_volume,
            state.current_phase,
            state.athlete.experience_level,
            recovery
        )

        # Generate next week's plan
        state = self._generate_weekly_plan(state)

        # Increment week counter
        state.week_number += 1
        state.weeks_in_current_phase += 1

        return state

    def _apply_deload(self, state: RunPlanState) -> RunPlanState:
        """
        Apply deload week.

        Reduces volume and intensity for recovery.

        Args:
            state: Current state

        Returns:
            Updated state with deload applied
        """
        state.status = PlanStatus.DELOAD

        # Reduce volume by 40%
        state.current_volume *= 0.60

        # Reduce frequency by 1
        state.current_frequency = max(2, state.current_frequency - 1)

        # Reset counters
        state.weeks_since_deload = 0
        state.consecutive_high_acwr_weeks = 0

        return state

    def _check_phase_transition(self, state: RunPlanState) -> RunPlanState:
        """
        Check if athlete is ready to transition to next phase.

        Args:
            state: Current state

        Returns:
            Updated state (possibly with new phase)
        """
        # Base → Transition criteria
        if state.current_phase == TrainingPhase.BASE_BUILDING:
            ready = (
                state.weeks_in_current_phase >= 6 and
                0.8 <= state.current_acwr <= 1.2 and
                state.current_volume >= 90  # Minimum volume
            )
            if ready:
                state.current_phase = TrainingPhase.TRANSITION
                state.weeks_in_current_phase = 0
                state.status = PlanStatus.TRANSITION

        # Transition → 80/20 criteria
        elif state.current_phase == TrainingPhase.TRANSITION:
            ready = (
                state.weeks_in_current_phase >= 2 and
                state.current_acwr <= 1.3
            )
            if ready:
                state.current_phase = TrainingPhase.EIGHTY_TWENTY
                state.weeks_in_current_phase = 0
                state.status = PlanStatus.ACTIVE

        return state

    def get_weekly_summary(self, state: RunPlanState) -> Dict[str, Any]:
        """
        Get summary of current week's plan.

        Args:
            state: Current state

        Returns:
            Dictionary with plan summary
        """
        summary = {
            'week_number': state.week_number,
            'phase': state.current_phase.value,
            'status': state.status.value,
            'volume': state.current_volume,
            'frequency': state.current_frequency,
            'acwr': state.current_acwr,
            'acwr_zone': state.acwr_zone,
            'maf_hr': state.maf_hr,
        }

        if state.current_workout_plan:
            summary['workout_plan'] = state.current_workout_plan.to_dict()

        if state.current_schedule:
            summary['schedule'] = state.current_schedule.to_dict()

        # Recent history
        if state.week_logs:
            recent = state.week_logs[-1]
            summary['last_week'] = {
                'planned_volume': recent.planned_volume,
                'actual_volume': recent.actual_volume,
                'adherence': recent.adherence_rate,
                'delta_v_applied': recent.delta_v_applied,
            }

        return summary

    def format_current_plan(self, state: RunPlanState) -> str:
        """
        Format current plan as readable text.

        Args:
            state: Current state

        Returns:
            Formatted string
        """
        lines = [
            f"Week {state.week_number} - {state.current_phase.value.replace('_', ' ').title()}",
            "=" * 50,
            f"Volume: {state.current_volume:.0f} min/week",
            f"Frequency: {state.current_frequency} days",
            f"ACWR: {state.current_acwr:.2f} ({state.acwr_zone})",
            f"MAF HR: {state.maf_hr} bpm",
            "",
        ]

        if state.current_schedule:
            lines.append(format_weekly_schedule(state.current_schedule))

        return "\n".join(lines)

    def simulate_progression(
        self,
        athlete: AthleteProfile,
        weeks: int = 12,
        compliance_rate: float = 0.9
    ) -> List[Dict[str, Any]]:
        """
        Simulate plan progression over multiple weeks.

        Useful for testing and visualization.

        Args:
            athlete: Athlete profile
            weeks: Number of weeks to simulate
            compliance_rate: Fraction of planned volume completed

        Returns:
            List of weekly summaries
        """
        state = self.initialize_plan(athlete)
        history = [self.get_weekly_summary(state)]

        np.random.seed(42)  # Reproducibility

        for week in range(weeks):
            # Simulate training with some variation
            actual_volume = state.current_volume * compliance_rate * np.random.normal(1, 0.1)
            actual_frequency = state.current_frequency

            # Generate synthetic TRIMP data
            daily_trimp = []
            for _ in range(7):
                # Training day
                if np.random.random() < state.current_frequency / 7:
                    duration = actual_volume / state.current_frequency
                    # Assume easy running at ~70% HR reserve
                    trimp = duration * 0.5 * 1.2  # Approximate TRIMP
                    daily_trimp.append(trimp)
                else:
                    daily_trimp.append(0)

            # Update plan
            state = self.update_with_training_data(
                state,
                daily_trimp,
                actual_volume,
                actual_frequency
            )

            history.append(self.get_weekly_summary(state))

        return history


def create_sample_athlete() -> AthleteProfile:
    """Create a sample athlete profile for testing."""
    return AthleteProfile(
        age=35,
        gender='male',
        weight_kg=80,
        height_cm=178,
        experience_level=ExperienceLevel.INTERMEDIATE,
        consistent_training_years=2,
        injury_free_years=1,
        health_status=HealthStatus.OPTIMAL,
        hr_rest=62,
        hr_max=185,
        recent_4wk_avg_volume=90,
    )


if __name__ == '__main__':
    print("Testing Run Plan Engine...")
    print("=" * 60)

    # Create engine and athlete
    engine = RunPlanEngine(use_optimized_params=True)
    athlete = create_sample_athlete()

    print(f"\nAthlete: {athlete.age}yo {athlete.gender}, BMI {athlete.bmi:.1f}")
    print(f"Experience: {athlete.experience_level.value}")

    # Initialize plan
    state = engine.initialize_plan(
        athlete,
        available_days={
            DayOfWeek.MONDAY, DayOfWeek.TUESDAY, DayOfWeek.WEDNESDAY,
            DayOfWeek.THURSDAY, DayOfWeek.SATURDAY, DayOfWeek.SUNDAY
        }
    )

    print("\n" + engine.format_current_plan(state))

    # Simulate 12 weeks
    print("\n" + "=" * 60)
    print("Simulating 12-week progression...")

    history = engine.simulate_progression(athlete, weeks=12)

    print("\nWeek | Phase          | Volume | Freq | ACWR  | Zone")
    print("-" * 60)
    for h in history:
        print(f"{h['week_number']:4d} | {h['phase']:14s} | {h['volume']:6.0f} | "
              f"{h['frequency']:4d} | {h['acwr']:5.2f} | {h['acwr_zone']}")

    # Calculate growth
    initial_vol = history[0]['volume']
    final_vol = history[-1]['volume']
    growth = final_vol / initial_vol

    print(f"\nVolume growth: {initial_vol:.0f} → {final_vol:.0f} min ({growth:.2f}x)")

    print("\n" + "=" * 60)
    print("All tests completed!")
