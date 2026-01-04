"""
Workout Types and Splits: Volume distribution across session types.

Based on:
- Fitzgerald, M. (2014). 80/20 Running
- Seiler, S. (2010). Polarized training model
- Maffetone, P. (2010). MAF training principles

These equations distribute weekly volume across workout types,
ensuring proper balance between easy, long, and hard sessions.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple
import math

from .frequency import TrainingPhase
from .maf import ExperienceLevel


class WorkoutType(Enum):
    """Classification of workout types."""
    EASY_SHORT = "easy_short"           # Short Zone 2 run
    EASY_LONG = "easy_long"             # Long Zone 2 run
    RECOVERY = "recovery"               # Very easy recovery run
    TEMPO = "tempo"                     # Sustained threshold effort
    INTERVALS = "intervals"             # High-intensity intervals
    FARTLEK = "fartlek"                 # Unstructured speed play
    HILL_REPEATS = "hill_repeats"       # Hill workout
    PROGRESSION = "progression"         # Negative split run
    RACE_PACE = "race_pace"             # Goal race pace work


class IntensityZone(Enum):
    """Heart rate intensity zones."""
    ZONE_1 = "zone_1"   # Recovery: < MAF - 10
    ZONE_2 = "zone_2"   # Aerobic: MAF - 10 to MAF
    ZONE_3 = "zone_3"   # Tempo: 83-87% HR max
    ZONE_4 = "zone_4"   # Threshold: 88-92% HR max
    ZONE_5 = "zone_5"   # VO2max: 93-100% HR max


@dataclass
class WorkoutSession:
    """
    A single workout session specification.

    Contains all information needed to execute and track a workout.
    """
    workout_type: WorkoutType
    duration_minutes: float
    intensity_zone: IntensityZone
    description: str = ""

    # For interval workouts
    interval_structure: Optional[str] = None  # e.g., "6 x 3 min @ Z4, 2 min recovery"

    # Heart rate targets (set based on athlete's zones)
    hr_target_low: Optional[int] = None
    hr_target_high: Optional[int] = None

    # RPE target
    rpe_target: float = 4.0  # 1-10 scale, 4 = easy

    # Stress score for scheduling
    stress_score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'workout_type': self.workout_type.value,
            'duration_minutes': self.duration_minutes,
            'intensity_zone': self.intensity_zone.value,
            'description': self.description,
            'interval_structure': self.interval_structure,
            'hr_target': (self.hr_target_low, self.hr_target_high),
            'rpe_target': self.rpe_target,
            'stress_score': self.stress_score,
        }


@dataclass
class WeeklyWorkoutPlan:
    """
    Complete weekly workout plan.

    Contains all sessions for the week with metadata.
    """
    sessions: List[WorkoutSession] = field(default_factory=list)
    phase: TrainingPhase = TrainingPhase.BASE_BUILDING
    total_volume: float = 0.0
    easy_percentage: float = 1.0
    hard_percentage: float = 0.0

    def add_session(self, session: WorkoutSession):
        """Add a session and update totals."""
        self.sessions.append(session)
        self.total_volume = sum(s.duration_minutes for s in self.sessions)
        self._recalculate_percentages()

    def _recalculate_percentages(self):
        """Recalculate easy/hard split."""
        if self.total_volume == 0:
            self.easy_percentage = 1.0
            self.hard_percentage = 0.0
            return

        easy_volume = sum(
            s.duration_minutes for s in self.sessions
            if s.intensity_zone in [IntensityZone.ZONE_1, IntensityZone.ZONE_2]
        )

        self.easy_percentage = easy_volume / self.total_volume
        self.hard_percentage = 1.0 - self.easy_percentage

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'sessions': [s.to_dict() for s in self.sessions],
            'phase': self.phase.value,
            'total_volume': self.total_volume,
            'easy_percentage': self.easy_percentage,
            'hard_percentage': self.hard_percentage,
            'session_count': len(self.sessions),
        }


def calculate_base_phase_splits(
    weekly_volume: float,
    frequency: int,
    experience_level: ExperienceLevel = ExperienceLevel.INTERMEDIATE
) -> WeeklyWorkoutPlan:
    """
    Calculate workout splits for base building phase.

    Base phase is 100% Zone 2 (MAF training).
    Distribution:
        - 70% volume across short easy runs
        - 30% volume in one long run

    Args:
        weekly_volume: Total weekly volume in minutes
        frequency: Number of running days
        experience_level: Runner's experience level

    Returns:
        WeeklyWorkoutPlan with all sessions
    """
    plan = WeeklyWorkoutPlan(phase=TrainingPhase.BASE_BUILDING)

    # Calculate split percentages
    long_run_pct = 0.30  # 30% of volume
    short_run_pct = 0.70  # 70% across remaining runs

    # Long run duration
    long_run_duration = weekly_volume * long_run_pct

    # Cap long run for beginners
    max_long_run = {
        ExperienceLevel.BEGINNER: 45,
        ExperienceLevel.NOVICE: 60,
        ExperienceLevel.INTERMEDIATE: 90,
        ExperienceLevel.ADVANCED: 120,
    }
    long_run_duration = min(long_run_duration, max_long_run.get(experience_level, 60))

    # Recalculate short run volume
    remaining_volume = weekly_volume - long_run_duration
    short_run_count = frequency - 1

    if short_run_count > 0:
        short_run_duration = remaining_volume / short_run_count
    else:
        short_run_duration = 0

    # Enforce minimum run duration
    min_run_duration = 20.0
    if short_run_duration < min_run_duration and short_run_count > 1:
        # Combine short runs
        short_run_count = max(1, int(remaining_volume / min_run_duration))
        short_run_duration = remaining_volume / short_run_count

    # Create sessions
    for i in range(short_run_count):
        session = WorkoutSession(
            workout_type=WorkoutType.EASY_SHORT,
            duration_minutes=round(short_run_duration, 0),
            intensity_zone=IntensityZone.ZONE_2,
            description=f"Easy run #{i+1} at MAF heart rate",
            rpe_target=4.0,
            stress_score=1.0,
        )
        plan.add_session(session)

    # Add long run
    if long_run_duration >= min_run_duration:
        long_session = WorkoutSession(
            workout_type=WorkoutType.EASY_LONG,
            duration_minutes=round(long_run_duration, 0),
            intensity_zone=IntensityZone.ZONE_2,
            description="Long easy run at MAF heart rate",
            rpe_target=4.0,
            stress_score=2.0,  # Higher stress due to duration
        )
        plan.add_session(long_session)

    return plan


def calculate_transition_splits(
    weekly_volume: float,
    frequency: int,
    experience_level: ExperienceLevel = ExperienceLevel.INTERMEDIATE
) -> WeeklyWorkoutPlan:
    """
    Calculate workout splits for transition phase.

    Transition introduces intensity gradually:
        - 90% easy (Zone 2)
        - 10% moderate (Zone 3 tempo)

    Args:
        weekly_volume: Total weekly volume in minutes
        frequency: Number of running days
        experience_level: Runner's experience level

    Returns:
        WeeklyWorkoutPlan with all sessions
    """
    plan = WeeklyWorkoutPlan(phase=TrainingPhase.TRANSITION)

    # Distribution
    easy_pct = 0.90
    tempo_pct = 0.10
    long_run_pct = 0.25

    easy_volume = weekly_volume * easy_pct
    tempo_volume = weekly_volume * tempo_pct
    long_run_duration = weekly_volume * long_run_pct

    # Tempo run (one session)
    tempo_session = WorkoutSession(
        workout_type=WorkoutType.TEMPO,
        duration_minutes=round(tempo_volume, 0),
        intensity_zone=IntensityZone.ZONE_3,
        description="Tempo run at 83-87% max HR",
        rpe_target=6.5,
        stress_score=2.5,
    )
    plan.add_session(tempo_session)

    # Long run
    long_session = WorkoutSession(
        workout_type=WorkoutType.EASY_LONG,
        duration_minutes=round(long_run_duration, 0),
        intensity_zone=IntensityZone.ZONE_2,
        description="Long easy run",
        rpe_target=4.0,
        stress_score=2.0,
    )
    plan.add_session(long_session)

    # Distribute remaining easy volume
    remaining_easy = easy_volume - long_run_duration
    short_run_count = frequency - 2  # Minus tempo and long

    if short_run_count > 0 and remaining_easy > 0:
        short_run_duration = remaining_easy / short_run_count
        for i in range(short_run_count):
            session = WorkoutSession(
                workout_type=WorkoutType.EASY_SHORT,
                duration_minutes=round(short_run_duration, 0),
                intensity_zone=IntensityZone.ZONE_2,
                description=f"Easy run #{i+1}",
                rpe_target=4.0,
                stress_score=1.0,
            )
            plan.add_session(session)

    return plan


def calculate_eighty_twenty_splits(
    weekly_volume: float,
    frequency: int,
    experience_level: ExperienceLevel = ExperienceLevel.INTERMEDIATE,
    hard_session_count: int = 1
) -> WeeklyWorkoutPlan:
    """
    Calculate workout splits for 80/20 polarized training phase.

    Distribution:
        - 80% easy (Zone 1-2)
        - 20% hard (Zone 4-5)

    Hard sessions alternate between tempo and intervals.

    Args:
        weekly_volume: Total weekly volume in minutes
        frequency: Number of running days
        experience_level: Runner's experience level
        hard_session_count: Number of hard sessions (1 or 2)

    Returns:
        WeeklyWorkoutPlan with all sessions
    """
    plan = WeeklyWorkoutPlan(phase=TrainingPhase.EIGHTY_TWENTY)

    # Distribution
    easy_volume = weekly_volume * 0.80
    hard_volume = weekly_volume * 0.20
    long_run_pct = 0.25  # 25% of total in long run

    long_run_duration = min(
        weekly_volume * long_run_pct,
        {
            ExperienceLevel.BEGINNER: 50,
            ExperienceLevel.NOVICE: 70,
            ExperienceLevel.INTERMEDIATE: 90,
            ExperienceLevel.ADVANCED: 120,
        }.get(experience_level, 70)
    )

    # Hard session(s)
    hard_per_session = hard_volume / hard_session_count

    # First hard session: intervals
    interval_session = WorkoutSession(
        workout_type=WorkoutType.INTERVALS,
        duration_minutes=round(hard_per_session, 0),
        intensity_zone=IntensityZone.ZONE_4,
        description="Interval workout",
        interval_structure=_generate_interval_structure(hard_per_session, experience_level),
        rpe_target=8.0,
        stress_score=3.0,
    )
    plan.add_session(interval_session)

    # Second hard session (if applicable): tempo
    if hard_session_count >= 2:
        tempo_session = WorkoutSession(
            workout_type=WorkoutType.TEMPO,
            duration_minutes=round(hard_per_session, 0),
            intensity_zone=IntensityZone.ZONE_3,
            description="Tempo run at threshold",
            rpe_target=7.0,
            stress_score=2.5,
        )
        plan.add_session(tempo_session)

    # Long run
    long_session = WorkoutSession(
        workout_type=WorkoutType.EASY_LONG,
        duration_minutes=round(long_run_duration, 0),
        intensity_zone=IntensityZone.ZONE_2,
        description="Long easy run",
        rpe_target=4.0,
        stress_score=2.0,
    )
    plan.add_session(long_session)

    # Distribute remaining easy volume
    remaining_easy = easy_volume - long_run_duration
    easy_run_count = frequency - hard_session_count - 1  # Minus hard and long

    if easy_run_count > 0 and remaining_easy > 0:
        easy_run_duration = remaining_easy / easy_run_count
        for i in range(easy_run_count):
            session = WorkoutSession(
                workout_type=WorkoutType.EASY_SHORT,
                duration_minutes=round(easy_run_duration, 0),
                intensity_zone=IntensityZone.ZONE_2,
                description=f"Easy run #{i+1}",
                rpe_target=4.0,
                stress_score=1.0,
            )
            plan.add_session(session)

    return plan


def _generate_interval_structure(
    duration: float,
    experience_level: ExperienceLevel
) -> str:
    """
    Generate appropriate interval structure based on duration and experience.

    Args:
        duration: Total interval workout duration in minutes
        experience_level: Runner's experience level

    Returns:
        String description of interval structure
    """
    # Work:Rest ratio and interval length by experience
    if experience_level == ExperienceLevel.BEGINNER:
        work_duration = 2  # minutes
        rest_duration = 2
        intensity = "Zone 4 (88-92% HR max)"
    elif experience_level == ExperienceLevel.NOVICE:
        work_duration = 3
        rest_duration = 2
        intensity = "Zone 4 (88-92% HR max)"
    elif experience_level == ExperienceLevel.INTERMEDIATE:
        work_duration = 4
        rest_duration = 2
        intensity = "Zone 4 (90-95% HR max)"
    else:  # Advanced
        work_duration = 5
        rest_duration = 2.5
        intensity = "Zone 4-5 (92-98% HR max)"

    # Calculate number of intervals
    # Account for warmup (10 min) and cooldown (5 min)
    usable_duration = max(10, duration - 15)
    interval_cycle = work_duration + rest_duration
    num_intervals = int(usable_duration / interval_cycle)
    num_intervals = max(3, min(8, num_intervals))

    return f"{num_intervals} x {work_duration} min @ {intensity}, {rest_duration} min recovery"


def calculate_splits(
    weekly_volume: float,
    frequency: int,
    phase: TrainingPhase,
    experience_level: ExperienceLevel = ExperienceLevel.INTERMEDIATE,
    hard_session_count: int = 1
) -> WeeklyWorkoutPlan:
    """
    Master function to calculate workout splits for any phase.

    Args:
        weekly_volume: Total weekly volume in minutes
        frequency: Number of running days
        phase: Current training phase
        experience_level: Runner's experience level
        hard_session_count: Number of hard sessions (for 80/20 phases)

    Returns:
        WeeklyWorkoutPlan with all sessions
    """
    if phase == TrainingPhase.BASE_BUILDING:
        return calculate_base_phase_splits(
            weekly_volume, frequency, experience_level
        )

    elif phase == TrainingPhase.TRANSITION:
        return calculate_transition_splits(
            weekly_volume, frequency, experience_level
        )

    elif phase in [TrainingPhase.EIGHTY_TWENTY, TrainingPhase.RACE_SPECIFIC]:
        return calculate_eighty_twenty_splits(
            weekly_volume, frequency, experience_level, hard_session_count
        )

    elif phase == TrainingPhase.TAPER:
        # Taper: reduced volume, maintain some intensity
        plan = calculate_eighty_twenty_splits(
            weekly_volume * 0.7,  # Reduced volume
            max(2, frequency - 1),
            experience_level,
            hard_session_count=1
        )
        plan.phase = TrainingPhase.TAPER
        return plan

    elif phase == TrainingPhase.RECOVERY:
        # Recovery: all easy, shorter runs
        plan = WeeklyWorkoutPlan(phase=TrainingPhase.RECOVERY)
        run_duration = weekly_volume / frequency

        for i in range(frequency):
            session = WorkoutSession(
                workout_type=WorkoutType.RECOVERY,
                duration_minutes=round(run_duration, 0),
                intensity_zone=IntensityZone.ZONE_1,
                description=f"Recovery run #{i+1}",
                rpe_target=3.0,
                stress_score=0.5,
            )
            plan.add_session(session)

        return plan

    else:
        # Default to base phase
        return calculate_base_phase_splits(
            weekly_volume, frequency, experience_level
        )


def get_workout_type_description(workout_type: WorkoutType) -> Dict[str, Any]:
    """
    Get detailed description of a workout type.

    Args:
        workout_type: The workout type

    Returns:
        Dictionary with workout details
    """
    descriptions = {
        WorkoutType.EASY_SHORT: {
            'name': 'Easy Short Run',
            'hr_zone': 'Zone 2 (MAF)',
            'rpe': '3-4',
            'description': 'Conversational pace run. Should feel comfortable throughout.',
            'purpose': 'Aerobic base building, recovery between hard efforts.',
        },
        WorkoutType.EASY_LONG: {
            'name': 'Long Easy Run',
            'hr_zone': 'Zone 2 (MAF)',
            'rpe': '3-4',
            'description': 'Extended duration at easy pace. Build endurance.',
            'purpose': 'Aerobic endurance, fat adaptation, mental preparation.',
        },
        WorkoutType.RECOVERY: {
            'name': 'Recovery Run',
            'hr_zone': 'Zone 1',
            'rpe': '2-3',
            'description': 'Very easy run, slower than normal easy pace.',
            'purpose': 'Active recovery, blood flow without additional stress.',
        },
        WorkoutType.TEMPO: {
            'name': 'Tempo Run',
            'hr_zone': 'Zone 3 (83-87% max HR)',
            'rpe': '6-7',
            'description': 'Sustained effort at "comfortably hard" pace.',
            'purpose': 'Lactate threshold improvement, race-pace preparation.',
        },
        WorkoutType.INTERVALS: {
            'name': 'Interval Workout',
            'hr_zone': 'Zone 4-5 (88-100% max HR)',
            'rpe': '8-9',
            'description': 'High-intensity efforts with recovery periods.',
            'purpose': 'VO2max development, speed and power.',
        },
        WorkoutType.FARTLEK: {
            'name': 'Fartlek',
            'hr_zone': 'Varies (Zone 2-4)',
            'rpe': '5-7',
            'description': 'Unstructured speed play with varied intensities.',
            'purpose': 'Aerobic capacity, mental engagement, fun.',
        },
        WorkoutType.HILL_REPEATS: {
            'name': 'Hill Repeats',
            'hr_zone': 'Zone 4-5',
            'rpe': '8-9',
            'description': 'High-effort uphill intervals with recovery jogs down.',
            'purpose': 'Strength, power, running economy.',
        },
        WorkoutType.PROGRESSION: {
            'name': 'Progression Run',
            'hr_zone': 'Zone 2 → Zone 3',
            'rpe': '4 → 7',
            'description': 'Run that gets progressively faster.',
            'purpose': 'Race simulation, finishing strong.',
        },
        WorkoutType.RACE_PACE: {
            'name': 'Race Pace Workout',
            'hr_zone': 'Goal race HR',
            'rpe': '6-8',
            'description': 'Practice at goal race effort.',
            'purpose': 'Race preparation, pacing practice.',
        },
    }

    return descriptions.get(workout_type, {
        'name': workout_type.value,
        'description': 'Standard running workout',
    })


if __name__ == '__main__':
    print("Testing Workout Splits Equations...")
    print("=" * 60)

    # Test base phase
    print("\n--- Base Phase (120 min, 4 days) ---")
    base_plan = calculate_base_phase_splits(
        120, 4, ExperienceLevel.INTERMEDIATE
    )
    print(f"Total sessions: {len(base_plan.sessions)}")
    for session in base_plan.sessions:
        print(f"  {session.workout_type.value}: {session.duration_minutes} min")
    print(f"Easy/Hard split: {base_plan.easy_percentage*100:.0f}/{base_plan.hard_percentage*100:.0f}")

    # Test transition phase
    print("\n--- Transition Phase (150 min, 4 days) ---")
    trans_plan = calculate_transition_splits(
        150, 4, ExperienceLevel.INTERMEDIATE
    )
    print(f"Total sessions: {len(trans_plan.sessions)}")
    for session in trans_plan.sessions:
        print(f"  {session.workout_type.value}: {session.duration_minutes} min ({session.intensity_zone.value})")
    print(f"Easy/Hard split: {trans_plan.easy_percentage*100:.0f}/{trans_plan.hard_percentage*100:.0f}")

    # Test 80/20 phase
    print("\n--- 80/20 Phase (180 min, 5 days, 2 hard) ---")
    eighty_plan = calculate_eighty_twenty_splits(
        180, 5, ExperienceLevel.INTERMEDIATE, hard_session_count=2
    )
    print(f"Total sessions: {len(eighty_plan.sessions)}")
    for session in eighty_plan.sessions:
        extra = f" [{session.interval_structure}]" if session.interval_structure else ""
        print(f"  {session.workout_type.value}: {session.duration_minutes} min ({session.intensity_zone.value}){extra}")
    print(f"Easy/Hard split: {eighty_plan.easy_percentage*100:.0f}/{eighty_plan.hard_percentage*100:.0f}")

    print("\n" + "=" * 60)
    print("All tests completed!")
