"""
Scheduling and Day Assignment: Optimal workout placement within a week.

Based on:
- Recovery principles from sports science
- Hard/easy sequencing research
- Constraint satisfaction for optimal scheduling

These equations determine the optimal placement of workouts within
a week, ensuring adequate recovery between hard efforts while
respecting athlete availability and preferences.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple, Set
from datetime import date, timedelta
import random

from .maf import ExperienceLevel
from .workout_splits import WorkoutSession, WorkoutType, WeeklyWorkoutPlan


class DayOfWeek(Enum):
    """Days of the week."""
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6


@dataclass
class DayPreference:
    """
    Athlete preferences for specific days.

    Higher scores indicate stronger preference.
    """
    day: DayOfWeek
    available: bool = True
    preference_score: float = 1.0  # 0-2 scale, 1 = neutral
    max_duration: Optional[float] = None  # Maximum run duration on this day
    preferred_workout_types: List[WorkoutType] = field(default_factory=list)


@dataclass
class ScheduledWorkout:
    """
    A workout assigned to a specific day.
    """
    session: WorkoutSession
    day: DayOfWeek
    day_of_week_name: str = ""

    def __post_init__(self):
        self.day_of_week_name = self.day.name.capitalize()


@dataclass
class WeekSchedule:
    """
    Complete weekly schedule with workouts assigned to days.
    """
    workouts: List[ScheduledWorkout] = field(default_factory=list)
    rest_days: List[DayOfWeek] = field(default_factory=list)
    week_start_date: Optional[date] = None

    def get_workout_for_day(self, day: DayOfWeek) -> Optional[ScheduledWorkout]:
        """Get workout scheduled for a specific day."""
        for workout in self.workouts:
            if workout.day == day:
                return workout
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        schedule = {}
        for day in DayOfWeek:
            workout = self.get_workout_for_day(day)
            if workout:
                schedule[day.name] = {
                    'workout_type': workout.session.workout_type.value,
                    'duration': workout.session.duration_minutes,
                    'intensity': workout.session.intensity_zone.value,
                    'description': workout.session.description,
                }
            else:
                schedule[day.name] = {'rest': True}

        return {
            'schedule': schedule,
            'workout_count': len(self.workouts),
            'rest_day_count': len(self.rest_days),
        }


def calculate_max_consecutive_runs(
    experience_level: ExperienceLevel,
    weekly_volume: float,
    bmi: Optional[float] = None
) -> int:
    """
    Calculate maximum consecutive running days allowed.

    Formula:
        max_consecutive = {
            2 if beginner or BMI > 30 or volume < 120
            3 if intermediate and volume 120-200
            4 if advanced and volume > 200
        }

    Args:
        experience_level: Runner's experience level
        weekly_volume: Weekly running volume in minutes
        bmi: Body Mass Index (optional)

    Returns:
        Maximum consecutive running days
    """
    # Base by experience
    if experience_level == ExperienceLevel.BEGINNER:
        base_max = 2
    elif experience_level == ExperienceLevel.NOVICE:
        base_max = 2
    elif experience_level == ExperienceLevel.INTERMEDIATE:
        base_max = 3
    else:  # Advanced
        base_max = 4

    # Reduce for low volume
    if weekly_volume < 120:
        base_max = min(base_max, 2)

    # Reduce for high BMI
    if bmi is not None and bmi > 30:
        base_max = min(base_max, 2)

    # Increase for high volume (advanced only)
    if experience_level == ExperienceLevel.ADVANCED and weekly_volume > 200:
        base_max = max(base_max, 4)

    return base_max


def calculate_min_days_between_hard(
    experience_level: ExperienceLevel,
    frequency: int
) -> int:
    """
    Calculate minimum days between hard workouts.

    Hard workouts require recovery time. This ensures adequate spacing.

    Args:
        experience_level: Runner's experience level
        frequency: Total running days per week

    Returns:
        Minimum days between hard sessions
    """
    if experience_level == ExperienceLevel.BEGINNER:
        return 3  # Very conservative
    elif experience_level == ExperienceLevel.NOVICE:
        return 2
    elif experience_level == ExperienceLevel.INTERMEDIATE:
        return 2 if frequency <= 4 else 1
    else:  # Advanced
        return 1 if frequency >= 5 else 2


def calculate_stress_score(session: WorkoutSession) -> float:
    """
    Calculate stress score for a workout session.

    Used for scheduling optimization to avoid clustering high-stress days.

    Args:
        session: The workout session

    Returns:
        Stress score (higher = more stressful)
    """
    # Use session's stress_score if available
    if hasattr(session, 'stress_score') and session.stress_score > 0:
        return session.stress_score

    # Default scores by workout type
    type_scores = {
        WorkoutType.RECOVERY: 0.5,
        WorkoutType.EASY_SHORT: 1.0,
        WorkoutType.EASY_LONG: 2.0,
        WorkoutType.FARTLEK: 2.0,
        WorkoutType.PROGRESSION: 2.5,
        WorkoutType.TEMPO: 2.5,
        WorkoutType.INTERVALS: 3.0,
        WorkoutType.HILL_REPEATS: 3.0,
        WorkoutType.RACE_PACE: 3.0,
    }

    base_score = type_scores.get(session.workout_type, 1.0)

    # Adjust for duration
    duration_factor = session.duration_minutes / 45.0  # Normalize to 45 min
    adjusted_score = base_score * (0.7 + 0.3 * duration_factor)

    return adjusted_score


def calculate_proximity_penalty(
    day: DayOfWeek,
    current_schedule: Dict[DayOfWeek, ScheduledWorkout],
    session: WorkoutSession
) -> float:
    """
    Calculate penalty for placing a workout adjacent to high-stress days.

    Args:
        day: Proposed day for the workout
        current_schedule: Already scheduled workouts
        session: Session to be scheduled

    Returns:
        Penalty score (higher = worse placement)
    """
    penalty = 0.0

    # Check adjacent days
    prev_day = DayOfWeek((day.value - 1) % 7)
    next_day = DayOfWeek((day.value + 1) % 7)

    for adjacent_day in [prev_day, next_day]:
        if adjacent_day in current_schedule:
            adjacent_workout = current_schedule[adjacent_day]
            adjacent_stress = calculate_stress_score(adjacent_workout.session)

            # High-stress adjacent = high penalty
            if adjacent_stress >= 2.5:
                penalty += 2.0
            elif adjacent_stress >= 2.0:
                penalty += 1.0

            # Special case: never place hard sessions adjacent to each other
            if (session.stress_score >= 2.5 and adjacent_stress >= 2.5):
                penalty += 5.0  # Strong deterrent

    return penalty


def is_valid_placement(
    day: DayOfWeek,
    session: WorkoutSession,
    current_schedule: Dict[DayOfWeek, ScheduledWorkout],
    max_consecutive: int,
    min_hard_spacing: int,
    available_days: Set[DayOfWeek]
) -> bool:
    """
    Check if placing a workout on a given day is valid.

    Args:
        day: Proposed day
        session: Session to place
        current_schedule: Already scheduled workouts
        max_consecutive: Maximum consecutive running days
        min_hard_spacing: Minimum days between hard sessions
        available_days: Days the athlete can run

    Returns:
        True if placement is valid
    """
    # Must be available
    if day not in available_days:
        return False

    # Day must not already have a workout
    if day in current_schedule:
        return False

    # Check consecutive days constraint
    consecutive_count = 1
    for offset in range(1, max_consecutive + 1):
        check_day = DayOfWeek((day.value - offset) % 7)
        if check_day in current_schedule:
            consecutive_count += 1
        else:
            break

    for offset in range(1, max_consecutive + 1):
        check_day = DayOfWeek((day.value + offset) % 7)
        if check_day in current_schedule:
            consecutive_count += 1
        else:
            break

    if consecutive_count > max_consecutive:
        return False

    # Check hard workout spacing
    if session.stress_score >= 2.5:  # Hard workout
        for offset in range(1, min_hard_spacing + 1):
            for direction in [-1, 1]:
                check_day = DayOfWeek((day.value + direction * offset) % 7)
                if check_day in current_schedule:
                    adjacent = current_schedule[check_day]
                    if adjacent.session.stress_score >= 2.5:
                        return False  # Another hard session too close

    return True


def get_day_preference_score(
    day: DayOfWeek,
    session: WorkoutSession,
    preferences: Optional[List[DayPreference]] = None
) -> float:
    """
    Get preference score for placing a session on a specific day.

    Considers both general preferences and workout-type-specific preferences.

    Args:
        day: The day to evaluate
        session: The session to place
        preferences: List of day preferences

    Returns:
        Preference score (higher = more preferred)
    """
    base_score = 1.0

    # Default preferences (if none specified)
    if preferences is None:
        # Long runs prefer weekends
        if session.workout_type == WorkoutType.EASY_LONG:
            if day in [DayOfWeek.SATURDAY, DayOfWeek.SUNDAY]:
                base_score += 0.5

        # Hard workouts prefer midweek (Tue/Wed/Thu)
        if session.stress_score >= 2.5:
            if day in [DayOfWeek.TUESDAY, DayOfWeek.WEDNESDAY, DayOfWeek.THURSDAY]:
                base_score += 0.3

        return base_score

    # Use explicit preferences
    for pref in preferences:
        if pref.day == day:
            if not pref.available:
                return -100  # Not available

            base_score = pref.preference_score

            # Check duration constraint
            if pref.max_duration and session.duration_minutes > pref.max_duration:
                base_score -= 1.0

            # Bonus for preferred workout types
            if session.workout_type in pref.preferred_workout_types:
                base_score += 0.5

            break

    return base_score


def assign_days(
    workout_plan: WeeklyWorkoutPlan,
    experience_level: ExperienceLevel,
    available_days: Optional[Set[DayOfWeek]] = None,
    preferences: Optional[List[DayPreference]] = None,
    bmi: Optional[float] = None,
    weekly_volume: Optional[float] = None
) -> WeekSchedule:
    """
    Assign workouts to specific days of the week.

    Uses constraint satisfaction with optimization to find the best schedule.

    Algorithm:
        1. Sort workouts by stress score (highest first)
        2. For each workout, find valid days
        3. Score each valid day (preference - proximity penalty)
        4. Assign to best-scoring day

    Args:
        workout_plan: Weekly workout plan with sessions
        experience_level: Runner's experience level
        available_days: Set of days athlete can run
        preferences: Day-specific preferences
        bmi: Body Mass Index (for consecutive day limit)
        weekly_volume: Weekly volume (for consecutive day limit)

    Returns:
        WeekSchedule with assigned days
    """
    # Default: all days available
    if available_days is None:
        available_days = set(DayOfWeek)

    # Calculate constraints
    volume = weekly_volume or workout_plan.total_volume
    max_consecutive = calculate_max_consecutive_runs(
        experience_level, volume, bmi
    )
    min_hard_spacing = calculate_min_days_between_hard(
        experience_level, len(workout_plan.sessions)
    )

    # Sort sessions by stress (highest first) to place hard workouts first
    sorted_sessions = sorted(
        workout_plan.sessions,
        key=lambda s: calculate_stress_score(s),
        reverse=True
    )

    schedule: Dict[DayOfWeek, ScheduledWorkout] = {}
    remaining_days = set(available_days)

    for session in sorted_sessions:
        best_day = None
        best_score = float('-inf')

        for day in remaining_days:
            # Check validity
            if not is_valid_placement(
                day, session, schedule, max_consecutive,
                min_hard_spacing, available_days
            ):
                continue

            # Calculate score
            preference = get_day_preference_score(day, session, preferences)
            proximity_penalty = calculate_proximity_penalty(day, schedule, session)

            score = preference - proximity_penalty

            if score > best_score:
                best_score = score
                best_day = day

        if best_day is not None:
            scheduled = ScheduledWorkout(session=session, day=best_day)
            schedule[best_day] = scheduled
            remaining_days.discard(best_day)
        else:
            # Fallback: try any remaining day
            for day in remaining_days:
                if day not in schedule:
                    scheduled = ScheduledWorkout(session=session, day=day)
                    schedule[best_day] = scheduled
                    remaining_days.discard(day)
                    break

    # Build final schedule
    week_schedule = WeekSchedule()
    for day in sorted(schedule.keys(), key=lambda d: d.value):
        week_schedule.workouts.append(schedule[day])

    # Identify rest days
    for day in DayOfWeek:
        if day not in schedule and day in available_days:
            week_schedule.rest_days.append(day)

    return week_schedule


def optimize_schedule(
    workout_plan: WeeklyWorkoutPlan,
    experience_level: ExperienceLevel,
    available_days: Optional[Set[DayOfWeek]] = None,
    preferences: Optional[List[DayPreference]] = None,
    bmi: Optional[float] = None,
    iterations: int = 100
) -> WeekSchedule:
    """
    Optimize schedule using multiple attempts.

    Tries different random orderings and returns the best schedule.

    Args:
        workout_plan: Weekly workout plan with sessions
        experience_level: Runner's experience level
        available_days: Set of days athlete can run
        preferences: Day-specific preferences
        bmi: Body Mass Index
        iterations: Number of optimization attempts

    Returns:
        Best WeekSchedule found
    """
    best_schedule = None
    best_score = float('-inf')

    for _ in range(iterations):
        # Shuffle sessions for variety
        shuffled_plan = WeeklyWorkoutPlan(
            phase=workout_plan.phase,
            total_volume=workout_plan.total_volume
        )
        sessions_copy = list(workout_plan.sessions)
        random.shuffle(sessions_copy)
        for s in sessions_copy:
            shuffled_plan.add_session(s)

        # Generate schedule
        schedule = assign_days(
            shuffled_plan,
            experience_level,
            available_days,
            preferences,
            bmi,
            workout_plan.total_volume
        )

        # Score schedule
        score = score_schedule(schedule, preferences)

        if score > best_score:
            best_score = score
            best_schedule = schedule

    return best_schedule or assign_days(
        workout_plan, experience_level, available_days, preferences, bmi
    )


def score_schedule(
    schedule: WeekSchedule,
    preferences: Optional[List[DayPreference]] = None
) -> float:
    """
    Score a schedule's quality.

    Higher score = better schedule.

    Args:
        schedule: The schedule to evaluate
        preferences: Day preferences

    Returns:
        Schedule quality score
    """
    score = 0.0

    for workout in schedule.workouts:
        # Preference score
        pref_score = get_day_preference_score(
            workout.day, workout.session, preferences
        )
        score += pref_score

        # Check stress clustering
        for other in schedule.workouts:
            if other.day == workout.day:
                continue

            day_diff = abs(workout.day.value - other.day.value)
            day_diff = min(day_diff, 7 - day_diff)  # Wrap around

            # Penalize high-stress days that are adjacent
            if day_diff == 1:
                combined_stress = (
                    calculate_stress_score(workout.session) +
                    calculate_stress_score(other.session)
                )
                if combined_stress > 4.0:
                    score -= 0.5

    # Bonus for well-distributed rest days
    if len(schedule.rest_days) >= 2:
        score += 0.5

    return score


def format_weekly_schedule(schedule: WeekSchedule) -> str:
    """
    Format schedule as a readable string.

    Args:
        schedule: The schedule to format

    Returns:
        Formatted string representation
    """
    lines = ["Weekly Schedule:", "=" * 40]

    for day in DayOfWeek:
        workout = schedule.get_workout_for_day(day)
        if workout:
            lines.append(
                f"{day.name:10s}: {workout.session.workout_type.value:15s} "
                f"({workout.session.duration_minutes:.0f} min)"
            )
        else:
            lines.append(f"{day.name:10s}: REST")

    return "\n".join(lines)


if __name__ == '__main__':
    from .workout_splits import calculate_eighty_twenty_splits

    print("Testing Scheduling Equations...")
    print("=" * 60)

    # Create a sample workout plan
    plan = calculate_eighty_twenty_splits(
        weekly_volume=180,
        frequency=5,
        experience_level=ExperienceLevel.INTERMEDIATE,
        hard_session_count=2
    )

    print(f"\nWorkout Plan ({len(plan.sessions)} sessions):")
    for session in plan.sessions:
        stress = calculate_stress_score(session)
        print(f"  {session.workout_type.value}: {session.duration_minutes} min (stress: {stress:.1f})")

    # Test day assignment
    print("\n--- Day Assignment ---")
    schedule = assign_days(
        plan,
        ExperienceLevel.INTERMEDIATE,
        available_days={
            DayOfWeek.MONDAY, DayOfWeek.TUESDAY, DayOfWeek.WEDNESDAY,
            DayOfWeek.THURSDAY, DayOfWeek.SATURDAY, DayOfWeek.SUNDAY
        },  # Friday off
        bmi=24.0
    )

    print(format_weekly_schedule(schedule))

    # Test with preferences
    print("\n--- With Preferences (long run Sunday) ---")
    preferences = [
        DayPreference(DayOfWeek.SUNDAY, preference_score=1.5,
                      preferred_workout_types=[WorkoutType.EASY_LONG]),
        DayPreference(DayOfWeek.FRIDAY, available=False),
    ]

    schedule2 = optimize_schedule(
        plan,
        ExperienceLevel.INTERMEDIATE,
        preferences=preferences,
        iterations=50
    )

    print(format_weekly_schedule(schedule2))

    # Test consecutive limits
    print("\n--- Consecutive Day Limits ---")
    for exp in ExperienceLevel:
        max_consec = calculate_max_consecutive_runs(exp, 150, bmi=25)
        print(f"  {exp.value}: max {max_consec} consecutive days")

    print("\n" + "=" * 60)
    print("All tests completed!")
