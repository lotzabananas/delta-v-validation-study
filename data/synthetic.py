"""
Synthetic runner data generation for backtesting.

Generates realistic runner profiles and training data with:
- Varied physiological characteristics (age, fitness, HR response)
- Day-to-day variance (fatigue, stress, illness)
- Compliance modeling (missed sessions)
- Natural adaptation curves
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import numpy as np
from core.metrics import calculate_trimp, estimate_maf_hr, estimate_hr_max


class FitnessLevel(Enum):
    """Fitness level classification."""
    SEDENTARY = 1      # No regular exercise
    BEGINNER = 2       # 0-6 months running
    RECREATIONAL = 3   # 6-24 months, casual runner
    INTERMEDIATE = 4   # 2-5 years, regular runner
    ADVANCED = 5       # 5+ years, competitive runner


class TrainingPhase(Enum):
    """Current training phase."""
    BASE_BUILDING = 1   # MAF/zone 2 focused
    TRANSITION = 2      # Moving to 80/20
    PERFORMANCE = 3     # Full 80/20 polarized


@dataclass
class RunnerProfile:
    """
    Complete runner profile for simulation.

    Includes physiological characteristics, current training status,
    and behavioral patterns.
    """
    # Identity
    id: str
    name: str

    # Demographics
    age: int
    gender: str  # 'male' or 'female'
    weight_kg: float

    # Fitness metrics
    fitness_level: FitnessLevel
    hr_rest: float
    hr_max: float
    maf_hr: float  # Zone 2 ceiling (180 - age + adjustment)

    # Current training status
    initial_weekly_volume: float  # minutes/week
    current_weekly_volume: float
    runs_per_week: int  # Typical frequency
    training_phase: TrainingPhase

    # Behavioral characteristics
    compliance_rate: float  # 0-1, probability of completing planned session
    intensity_discipline: float  # 0-1, adherence to target HR zones
    recovery_quality: float  # 0-1, sleep/nutrition quality

    # Adaptation characteristics
    adaptation_rate: float  # How quickly fitness improves (affects HR response)
    injury_susceptibility: float  # 0-1, sensitivity to overload

    # History tracking
    trimp_history: List[float] = field(default_factory=list)
    volume_history: List[float] = field(default_factory=list)
    acwr_history: List[float] = field(default_factory=list)

    def get_hr_for_zone2(self, day_variance: float = 0.0) -> float:
        """
        Get typical heart rate for zone 2 training.

        Args:
            day_variance: Random factor for daily variation (-1 to 1)

        Returns:
            Average HR for session
        """
        # Base zone 2 HR is 85-95% of MAF HR
        base_hr = self.maf_hr * 0.90
        # Add variance (±5 bpm typical)
        variance = day_variance * 5
        return np.clip(base_hr + variance, self.hr_rest + 20, self.maf_hr)

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'age': self.age,
            'gender': self.gender,
            'weight_kg': self.weight_kg,
            'fitness_level': self.fitness_level.name,
            'hr_rest': self.hr_rest,
            'hr_max': self.hr_max,
            'maf_hr': self.maf_hr,
            'initial_weekly_volume': self.initial_weekly_volume,
            'runs_per_week': self.runs_per_week,
            'compliance_rate': self.compliance_rate,
            'adaptation_rate': self.adaptation_rate,
            'injury_susceptibility': self.injury_susceptibility,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# RUNNER ARCHETYPES
# ═══════════════════════════════════════════════════════════════════════════════

def create_beginner_sedentary(id_num: int) -> RunnerProfile:
    """Sedentary beginner just starting to run."""
    age = np.random.randint(28, 45)
    gender = np.random.choice(['male', 'female'])

    return RunnerProfile(
        id=f"beginner_sedentary_{id_num}",
        name=f"Beginner Sedentary {id_num}",
        age=age,
        gender=gender,
        weight_kg=np.random.uniform(70, 95) if gender == 'male' else np.random.uniform(60, 80),
        fitness_level=FitnessLevel.SEDENTARY,
        hr_rest=np.random.uniform(68, 80),
        hr_max=estimate_hr_max(age),
        maf_hr=estimate_maf_hr(age, adjustment=-5),  # Conservative for beginners
        initial_weekly_volume=np.random.uniform(45, 75),
        current_weekly_volume=np.random.uniform(45, 75),
        runs_per_week=3,
        training_phase=TrainingPhase.BASE_BUILDING,
        compliance_rate=np.random.uniform(0.75, 0.90),
        intensity_discipline=np.random.uniform(0.60, 0.80),
        recovery_quality=np.random.uniform(0.50, 0.70),
        adaptation_rate=np.random.uniform(0.03, 0.05),  # Faster newbie gains
        injury_susceptibility=np.random.uniform(0.50, 0.70),  # Higher risk
    )


def create_returning_runner(id_num: int) -> RunnerProfile:
    """Runner returning after break, has base fitness."""
    age = np.random.randint(30, 50)
    gender = np.random.choice(['male', 'female'])

    return RunnerProfile(
        id=f"returning_{id_num}",
        name=f"Returning Runner {id_num}",
        age=age,
        gender=gender,
        weight_kg=np.random.uniform(65, 85) if gender == 'male' else np.random.uniform(55, 75),
        fitness_level=FitnessLevel.RECREATIONAL,
        hr_rest=np.random.uniform(58, 68),
        hr_max=estimate_hr_max(age),
        maf_hr=estimate_maf_hr(age, adjustment=0),
        initial_weekly_volume=np.random.uniform(90, 150),
        current_weekly_volume=np.random.uniform(90, 150),
        runs_per_week=4,
        training_phase=TrainingPhase.BASE_BUILDING,
        compliance_rate=np.random.uniform(0.80, 0.95),
        intensity_discipline=np.random.uniform(0.70, 0.85),
        recovery_quality=np.random.uniform(0.60, 0.80),
        adaptation_rate=np.random.uniform(0.02, 0.04),
        injury_susceptibility=np.random.uniform(0.35, 0.55),
    )


def create_young_athlete(id_num: int) -> RunnerProfile:
    """Young, fit athlete with good training capacity."""
    age = np.random.randint(18, 28)
    gender = np.random.choice(['male', 'female'])

    return RunnerProfile(
        id=f"young_athlete_{id_num}",
        name=f"Young Athlete {id_num}",
        age=age,
        gender=gender,
        weight_kg=np.random.uniform(60, 78) if gender == 'male' else np.random.uniform(50, 65),
        fitness_level=FitnessLevel.INTERMEDIATE,
        hr_rest=np.random.uniform(50, 60),
        hr_max=estimate_hr_max(age),
        maf_hr=estimate_maf_hr(age, adjustment=5),  # Can push harder
        initial_weekly_volume=np.random.uniform(120, 200),
        current_weekly_volume=np.random.uniform(120, 200),
        runs_per_week=5,
        training_phase=TrainingPhase.BASE_BUILDING,
        compliance_rate=np.random.uniform(0.85, 0.98),
        intensity_discipline=np.random.uniform(0.75, 0.90),
        recovery_quality=np.random.uniform(0.70, 0.90),
        adaptation_rate=np.random.uniform(0.025, 0.045),
        injury_susceptibility=np.random.uniform(0.25, 0.45),
    )


def create_masters_runner(id_num: int) -> RunnerProfile:
    """Masters runner (50+), experienced but needs conservative approach."""
    age = np.random.randint(50, 65)
    gender = np.random.choice(['male', 'female'])

    return RunnerProfile(
        id=f"masters_{id_num}",
        name=f"Masters Runner {id_num}",
        age=age,
        gender=gender,
        weight_kg=np.random.uniform(68, 88) if gender == 'male' else np.random.uniform(55, 75),
        fitness_level=FitnessLevel.RECREATIONAL,
        hr_rest=np.random.uniform(55, 68),
        hr_max=estimate_hr_max(age),
        maf_hr=estimate_maf_hr(age, adjustment=-5),
        initial_weekly_volume=np.random.uniform(75, 150),
        current_weekly_volume=np.random.uniform(75, 150),
        runs_per_week=4,
        training_phase=TrainingPhase.BASE_BUILDING,
        compliance_rate=np.random.uniform(0.85, 0.95),
        intensity_discipline=np.random.uniform(0.80, 0.95),  # More disciplined
        recovery_quality=np.random.uniform(0.55, 0.75),  # Recovery slower
        adaptation_rate=np.random.uniform(0.015, 0.030),  # Slower adaptation
        injury_susceptibility=np.random.uniform(0.45, 0.65),
    )


def create_overweight_beginner(id_num: int) -> RunnerProfile:
    """Overweight beginner needing very conservative progression."""
    age = np.random.randint(30, 50)
    gender = np.random.choice(['male', 'female'])
    # BMI 28-35 range
    height_m = np.random.uniform(1.65, 1.85) if gender == 'male' else np.random.uniform(1.55, 1.75)
    bmi = np.random.uniform(28, 35)
    weight_kg = bmi * (height_m ** 2)

    return RunnerProfile(
        id=f"overweight_beginner_{id_num}",
        name=f"Overweight Beginner {id_num}",
        age=age,
        gender=gender,
        weight_kg=weight_kg,
        fitness_level=FitnessLevel.SEDENTARY,
        hr_rest=np.random.uniform(72, 85),
        hr_max=estimate_hr_max(age),
        maf_hr=estimate_maf_hr(age, adjustment=-10),  # Very conservative
        initial_weekly_volume=np.random.uniform(30, 60),
        current_weekly_volume=np.random.uniform(30, 60),
        runs_per_week=3,
        training_phase=TrainingPhase.BASE_BUILDING,
        compliance_rate=np.random.uniform(0.65, 0.85),
        intensity_discipline=np.random.uniform(0.50, 0.70),
        recovery_quality=np.random.uniform(0.45, 0.65),
        adaptation_rate=np.random.uniform(0.02, 0.04),
        injury_susceptibility=np.random.uniform(0.60, 0.80),  # Higher risk
    )


def create_experienced_recreational(id_num: int) -> RunnerProfile:
    """Experienced recreational runner with solid base."""
    age = np.random.randint(28, 45)
    gender = np.random.choice(['male', 'female'])

    return RunnerProfile(
        id=f"experienced_rec_{id_num}",
        name=f"Experienced Recreational {id_num}",
        age=age,
        gender=gender,
        weight_kg=np.random.uniform(62, 80) if gender == 'male' else np.random.uniform(52, 68),
        fitness_level=FitnessLevel.INTERMEDIATE,
        hr_rest=np.random.uniform(52, 62),
        hr_max=estimate_hr_max(age),
        maf_hr=estimate_maf_hr(age, adjustment=0),
        initial_weekly_volume=np.random.uniform(150, 240),
        current_weekly_volume=np.random.uniform(150, 240),
        runs_per_week=5,
        training_phase=TrainingPhase.BASE_BUILDING,
        compliance_rate=np.random.uniform(0.85, 0.95),
        intensity_discipline=np.random.uniform(0.75, 0.90),
        recovery_quality=np.random.uniform(0.65, 0.85),
        adaptation_rate=np.random.uniform(0.015, 0.030),
        injury_susceptibility=np.random.uniform(0.30, 0.50),
    )


def create_injury_prone(id_num: int) -> RunnerProfile:
    """Runner with history of injuries, needs extra conservative approach."""
    age = np.random.randint(30, 50)
    gender = np.random.choice(['male', 'female'])

    return RunnerProfile(
        id=f"injury_prone_{id_num}",
        name=f"Injury Prone Runner {id_num}",
        age=age,
        gender=gender,
        weight_kg=np.random.uniform(65, 85) if gender == 'male' else np.random.uniform(55, 72),
        fitness_level=FitnessLevel.RECREATIONAL,
        hr_rest=np.random.uniform(58, 68),
        hr_max=estimate_hr_max(age),
        maf_hr=estimate_maf_hr(age, adjustment=-5),
        initial_weekly_volume=np.random.uniform(60, 120),
        current_weekly_volume=np.random.uniform(60, 120),
        runs_per_week=4,
        training_phase=TrainingPhase.BASE_BUILDING,
        compliance_rate=np.random.uniform(0.75, 0.90),
        intensity_discipline=np.random.uniform(0.70, 0.85),
        recovery_quality=np.random.uniform(0.55, 0.75),
        adaptation_rate=np.random.uniform(0.015, 0.025),
        injury_susceptibility=np.random.uniform(0.70, 0.90),  # Very high
    )


def create_eager_overreacher(id_num: int) -> RunnerProfile:
    """Eager runner who tends to do too much too soon."""
    age = np.random.randint(25, 40)
    gender = np.random.choice(['male', 'female'])

    return RunnerProfile(
        id=f"eager_{id_num}",
        name=f"Eager Overreacher {id_num}",
        age=age,
        gender=gender,
        weight_kg=np.random.uniform(65, 82) if gender == 'male' else np.random.uniform(55, 70),
        fitness_level=FitnessLevel.BEGINNER,
        hr_rest=np.random.uniform(62, 72),
        hr_max=estimate_hr_max(age),
        maf_hr=estimate_maf_hr(age, adjustment=0),
        initial_weekly_volume=np.random.uniform(75, 120),
        current_weekly_volume=np.random.uniform(75, 120),
        runs_per_week=4,
        training_phase=TrainingPhase.BASE_BUILDING,
        compliance_rate=np.random.uniform(0.90, 0.99),  # Very eager
        intensity_discipline=np.random.uniform(0.40, 0.60),  # Often goes too hard
        recovery_quality=np.random.uniform(0.55, 0.75),
        adaptation_rate=np.random.uniform(0.025, 0.040),
        injury_susceptibility=np.random.uniform(0.45, 0.65),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PROFILE GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

ARCHETYPE_CREATORS = [
    (create_beginner_sedentary, 3),      # 3 instances
    (create_returning_runner, 3),
    (create_young_athlete, 2),
    (create_masters_runner, 3),
    (create_overweight_beginner, 2),
    (create_experienced_recreational, 3),
    (create_injury_prone, 2),
    (create_eager_overreacher, 2),
]


def generate_runner_profiles(
    n_profiles: int = 20,
    seed: Optional[int] = None
) -> List[RunnerProfile]:
    """
    Generate diverse runner profiles for backtesting.

    Args:
        n_profiles: Number of profiles to generate
        seed: Random seed for reproducibility

    Returns:
        List of RunnerProfile objects
    """
    if seed is not None:
        np.random.seed(seed)

    profiles = []

    # First, create one of each archetype
    for creator, default_count in ARCHETYPE_CREATORS:
        for i in range(default_count):
            if len(profiles) >= n_profiles:
                break
            profiles.append(creator(i + 1))

    # If we need more, randomly select archetypes
    while len(profiles) < n_profiles:
        creator, _ = ARCHETYPE_CREATORS[np.random.randint(len(ARCHETYPE_CREATORS))]
        idx = len(profiles) + 1
        profiles.append(creator(idx))

    return profiles[:n_profiles]


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DailyTraining:
    """Single day of training data."""
    day: int
    duration_min: float
    hr_avg: float
    trimp: float
    completed: bool
    notes: str = ""


def generate_training_day(
    profile: RunnerProfile,
    target_duration: float,
    day_of_week: int,
    stress_level: float = 0.0,
    illness_factor: float = 0.0,
) -> DailyTraining:
    """
    Generate a single day's training data.

    Args:
        profile: Runner profile
        target_duration: Planned duration (minutes)
        day_of_week: 0=Monday, 6=Sunday
        stress_level: Life stress factor (0-1)
        illness_factor: Illness severity (0-1, 0=healthy)

    Returns:
        DailyTraining object
    """
    # Check compliance (did they run?)
    compliance_modifier = 1.0 - stress_level * 0.3 - illness_factor * 0.8
    actual_compliance = profile.compliance_rate * compliance_modifier

    if np.random.random() > actual_compliance:
        # Skipped session
        return DailyTraining(
            day=day_of_week,
            duration_min=0,
            hr_avg=0,
            trimp=0,
            completed=False,
            notes="Skipped"
        )

    # Duration variance (most people run a bit more or less than planned)
    duration_variance = np.random.normal(0, 0.10)  # ±10% typical
    if illness_factor > 0:
        duration_variance -= illness_factor * 0.3  # Cut short if ill
    actual_duration = target_duration * (1 + duration_variance)
    actual_duration = max(10, actual_duration)  # Minimum 10 min

    # Heart rate for zone 2 training
    # Poor discipline = higher HR, illness = elevated HR
    hr_base = profile.get_hr_for_zone2()
    hr_elevation = (1 - profile.intensity_discipline) * np.random.uniform(0, 10)
    hr_illness = illness_factor * 10  # Elevated HR when sick
    hr_avg = hr_base + hr_elevation + hr_illness
    hr_avg = min(hr_avg, profile.hr_max - 10)  # Cap below max

    # Calculate TRIMP
    trimp = calculate_trimp(
        duration_min=actual_duration,
        hr_avg=hr_avg,
        hr_rest=profile.hr_rest,
        hr_max=profile.hr_max,
        gender=profile.gender
    )

    return DailyTraining(
        day=day_of_week,
        duration_min=actual_duration,
        hr_avg=hr_avg,
        trimp=trimp,
        completed=True,
        notes="Illness" if illness_factor > 0.3 else ""
    )


def generate_training_week(
    profile: RunnerProfile,
    target_weekly_volume: float,
    week_number: int,
    life_events: Optional[Dict[str, float]] = None,
) -> Tuple[List[DailyTraining], float]:
    """
    Generate a week of training data.

    Args:
        profile: Runner profile
        target_weekly_volume: Target volume for week (minutes)
        week_number: Week number in training block
        life_events: Optional dict with 'stress' and 'illness' factors

    Returns:
        Tuple of (list of daily training, total weekly TRIMP)
    """
    if life_events is None:
        life_events = {}

    stress = life_events.get('stress', np.random.uniform(0, 0.3))
    illness = life_events.get('illness', 0)

    # Distribute volume across runs_per_week sessions
    # Typical pattern: longer weekend run, shorter weekday runs
    runs = profile.runs_per_week
    per_session = target_weekly_volume / runs

    # Example schedule: Tu, Th, Sat, Sun for 4 runs/week
    run_days = _get_run_schedule(runs)

    week_data = []
    total_trimp = 0

    for day in range(7):
        if day in run_days:
            # Vary session length (80-120% of average)
            session_target = per_session * np.random.uniform(0.85, 1.20)
            # Weekend long run
            if day >= 5:
                session_target *= 1.2

            training = generate_training_day(
                profile, session_target, day,
                stress_level=stress,
                illness_factor=illness
            )
        else:
            # Rest day
            training = DailyTraining(
                day=day, duration_min=0, hr_avg=0,
                trimp=0, completed=True, notes="Rest"
            )

        week_data.append(training)
        total_trimp += training.trimp

    return week_data, total_trimp


def _get_run_schedule(runs_per_week: int) -> List[int]:
    """Get typical run days for given frequency."""
    schedules = {
        2: [2, 5],           # Wed, Sat
        3: [1, 3, 5],        # Tue, Thu, Sat
        4: [1, 3, 5, 6],     # Tue, Thu, Sat, Sun
        5: [0, 1, 3, 5, 6],  # Mon, Tue, Thu, Sat, Sun
        6: [0, 1, 2, 4, 5, 6],
        7: list(range(7)),
    }
    return schedules.get(runs_per_week, [1, 3, 5])  # Default 3x


def generate_warmup_period(
    profile: RunnerProfile,
    weeks: int = 4
) -> np.ndarray:
    """
    Generate warmup period to establish chronic load baseline.

    Args:
        profile: Runner profile
        weeks: Number of warmup weeks

    Returns:
        Array of daily TRIMP values for warmup period
    """
    all_trimps = []

    for week in range(weeks):
        week_data, _ = generate_training_week(
            profile,
            profile.initial_weekly_volume,
            week
        )
        for day in week_data:
            all_trimps.append(day.trimp)

    return np.array(all_trimps)


def inject_life_events(
    num_weeks: int,
    seed: Optional[int] = None
) -> List[Dict[str, float]]:
    """
    Generate realistic life events (stress, illness) over simulation.

    Args:
        num_weeks: Number of weeks in simulation
        seed: Random seed

    Returns:
        List of event dictionaries per week
    """
    if seed is not None:
        np.random.seed(seed)

    events = []
    for week in range(num_weeks):
        # Base stress varies
        stress = np.random.uniform(0, 0.3)

        # Occasional high stress weeks (15% chance)
        if np.random.random() < 0.15:
            stress = np.random.uniform(0.5, 0.8)

        # Illness (5% chance any given week)
        illness = 0.0
        if np.random.random() < 0.05:
            illness = np.random.uniform(0.3, 0.8)

        events.append({'stress': stress, 'illness': illness})

    return events


if __name__ == '__main__':
    print("Testing synthetic data generation...")

    # Generate profiles
    profiles = generate_runner_profiles(5, seed=42)
    print(f"\nGenerated {len(profiles)} profiles:")
    for p in profiles:
        print(f"  {p.name}: {p.age}yo {p.gender}, {p.fitness_level.name}, "
              f"Vol: {p.initial_weekly_volume:.0f} min/wk")

    # Generate a week of training for first profile
    profile = profiles[0]
    print(f"\nGenerating week for {profile.name}:")
    week_data, weekly_trimp = generate_training_week(
        profile,
        profile.initial_weekly_volume,
        week_number=1
    )

    print("Day | Duration | HR Avg | TRIMP | Status")
    print("-" * 50)
    for d in week_data:
        status = "Completed" if d.completed and d.duration_min > 0 else "Rest/Skip"
        print(f"{d.day:3d} | {d.duration_min:8.1f} | {d.hr_avg:6.1f} | {d.trimp:5.1f} | {status}")
    print(f"Weekly TRIMP: {weekly_trimp:.1f}")

    # Generate warmup period
    warmup_trimps = generate_warmup_period(profile, weeks=4)
    print(f"\nWarmup period: {len(warmup_trimps)} days, total TRIMP: {warmup_trimps.sum():.1f}")

    print("\nAll tests passed!")
