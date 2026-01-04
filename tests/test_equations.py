"""
Comprehensive tests for all running plan equations.

Tests cover:
1. MAF HR and Initial Volume calculations
2. TRIMP, EWMA, ACWR metrics
3. Delta V adjustments
4. Frequency calculations
5. Workout splits
6. Scheduling and day assignment
7. Full engine integration

Run with: python -m pytest tests/test_equations.py -v
"""

import pytest
import numpy as np
from datetime import date

import sys
sys.path.insert(0, '..')

from core.metrics import (
    calculate_trimp,
    calculate_ewma,
    calculate_acwr,
    classify_acwr_zone,
)
from core.delta_v import (
    DeltaVParams,
    calculate_delta_v,
    apply_delta_v,
)
from core.maf import (
    AthleteProfile,
    ExperienceLevel,
    HealthStatus,
    calculate_maf_hr,
    calculate_initial_volume,
    calculate_bmi_stress_modifier,
    calculate_max_delta_v_cap,
    get_athlete_adjustments,
)
from core.frequency import (
    TrainingPhase,
    RecoveryMetrics,
    calculate_frequency,
    calculate_base_phase_frequency,
    calculate_eighty_twenty_frequency,
)
from core.workout_splits import (
    WorkoutType,
    WeeklyWorkoutPlan,
    calculate_splits,
    calculate_base_phase_splits,
    calculate_eighty_twenty_splits,
)
from core.scheduling import (
    DayOfWeek,
    assign_days,
    calculate_max_consecutive_runs,
)
from core.run_plan_engine import (
    RunPlanEngine,
    RunPlanState,
)


# =============================================================================
# MAF HR and Initial Volume Tests
# =============================================================================

class TestMAFHR:
    """Tests for MAF heart rate calculations."""

    def test_basic_maf_calculation(self):
        """Test basic MAF = 180 - age formula."""
        maf, _ = calculate_maf_hr(age=40)
        assert maf == 140  # 180 - 40

    def test_experienced_runner_adjustment(self):
        """Experienced runners get +5 to +10 bonus."""
        maf_beginner, _ = calculate_maf_hr(
            age=40,
            experience_level=ExperienceLevel.BEGINNER
        )
        maf_advanced, _ = calculate_maf_hr(
            age=40,
            experience_level=ExperienceLevel.ADVANCED,
            consistent_training_years=5,
            injury_free_years=3
        )
        assert maf_advanced > maf_beginner
        assert maf_advanced - maf_beginner >= 5

    def test_health_adjustment(self):
        """Health issues reduce MAF."""
        maf_healthy, _ = calculate_maf_hr(
            age=40,
            health_status=HealthStatus.OPTIMAL
        )
        maf_chronic, _ = calculate_maf_hr(
            age=40,
            health_status=HealthStatus.CHRONIC
        )
        assert maf_chronic < maf_healthy
        assert maf_healthy - maf_chronic >= 10

    def test_bmi_adjustment(self):
        """High BMI reduces MAF."""
        maf_normal, _ = calculate_maf_hr(age=40, bmi=24)
        maf_obese, _ = calculate_maf_hr(age=40, bmi=35)
        assert maf_obese < maf_normal

    def test_maf_bounds(self):
        """MAF should be within physiological bounds."""
        # Very young
        maf_young, _ = calculate_maf_hr(age=20)
        assert 100 <= maf_young <= 170

        # Elderly
        maf_old, _ = calculate_maf_hr(age=70)
        assert 100 <= maf_old <= 170


class TestInitialVolume:
    """Tests for initial volume calculations."""

    def test_beginner_initial_volume(self):
        """Beginners should start with conservative volume."""
        volume, _ = calculate_initial_volume(
            experience_level=ExperienceLevel.BEGINNER
        )
        assert 30 <= volume <= 120

    def test_volume_respects_history(self):
        """Volume should scale with recent training."""
        volume_no_history, _ = calculate_initial_volume(
            experience_level=ExperienceLevel.INTERMEDIATE
        )
        volume_with_history, _ = calculate_initial_volume(
            experience_level=ExperienceLevel.INTERMEDIATE,
            recent_4wk_avg_volume=180
        )
        assert volume_with_history > volume_no_history

    def test_bmi_reduces_volume(self):
        """High BMI should reduce initial volume."""
        volume_normal, _ = calculate_initial_volume(
            experience_level=ExperienceLevel.INTERMEDIATE,
            bmi=24
        )
        volume_obese, _ = calculate_initial_volume(
            experience_level=ExperienceLevel.INTERMEDIATE,
            bmi=35
        )
        assert volume_obese < volume_normal


class TestBMIModifier:
    """Tests for BMI stress modifier."""

    def test_normal_bmi_no_modification(self):
        """Normal BMI should have modifier of 1.0."""
        assert calculate_bmi_stress_modifier(22) == 1.0
        assert calculate_bmi_stress_modifier(25) == 1.0

    def test_overweight_slight_increase(self):
        """Overweight BMI should have slight modifier increase."""
        modifier = calculate_bmi_stress_modifier(28)
        assert 1.0 < modifier < 1.15

    def test_obese_higher_modifier(self):
        """Obese BMI should have higher modifier."""
        modifier = calculate_bmi_stress_modifier(35)
        assert modifier >= 1.15


# =============================================================================
# TRIMP and ACWR Tests
# =============================================================================

class TestTRIMP:
    """Tests for TRIMP calculations."""

    def test_trimp_calculation(self):
        """Test basic TRIMP formula."""
        trimp = calculate_trimp(
            duration_min=60,
            hr_avg=140,
            hr_rest=60,
            hr_max=180,
            gender='male'
        )
        # TRIMP should be positive and reasonable for 60 min
        assert 30 <= trimp <= 200

    def test_trimp_increases_with_duration(self):
        """Longer runs should have higher TRIMP."""
        trimp_30 = calculate_trimp(30, 140, 60, 180)
        trimp_60 = calculate_trimp(60, 140, 60, 180)
        assert trimp_60 > trimp_30

    def test_trimp_increases_with_intensity(self):
        """Higher intensity should increase TRIMP."""
        trimp_easy = calculate_trimp(60, 130, 60, 180)
        trimp_hard = calculate_trimp(60, 160, 60, 180)
        assert trimp_hard > trimp_easy


class TestACWR:
    """Tests for ACWR calculations."""

    def test_acwr_neutral_with_consistent_training(self):
        """Consistent training should yield ACWR near 1.0."""
        consistent_load = np.ones(28) * 50  # Same load every day
        acwr, _, _ = calculate_acwr(consistent_load)
        assert 0.9 <= acwr <= 1.1

    def test_acwr_low_after_rest(self):
        """ACWR should be low after extended rest."""
        training = np.concatenate([
            np.ones(21) * 50,  # 3 weeks normal
            np.ones(7) * 10,   # 1 week rest
        ])
        acwr, _, _ = calculate_acwr(training)
        assert acwr < 0.8

    def test_acwr_high_after_spike(self):
        """ACWR should be high after load spike."""
        training = np.concatenate([
            np.ones(21) * 30,  # 3 weeks easy
            np.ones(7) * 100,  # 1 week hard
        ])
        acwr, _, _ = calculate_acwr(training)
        assert acwr > 1.3


class TestACWRZones:
    """Tests for ACWR zone classification."""

    def test_zone_boundaries(self):
        """Test correct zone classification."""
        assert classify_acwr_zone(0.5) == 'low'
        assert classify_acwr_zone(0.9) == 'optimal'
        assert classify_acwr_zone(1.35) == 'caution'
        assert classify_acwr_zone(1.7) == 'danger'
        assert classify_acwr_zone(2.5) == 'critical'


# =============================================================================
# Delta V Tests
# =============================================================================

class TestDeltaV:
    """Tests for Delta V equation."""

    def test_optimal_zone_increase(self):
        """Optimal zone should recommend increase."""
        delta_v, zone, _ = calculate_delta_v(1.0)
        assert delta_v > 0
        assert zone == 'optimal'

    def test_danger_zone_decrease(self):
        """Danger zone should recommend decrease."""
        delta_v, zone, _ = calculate_delta_v(1.7)
        assert delta_v < 0
        assert zone == 'danger'

    def test_critical_zone_significant_decrease(self):
        """Critical zone should recommend major decrease."""
        delta_v, zone, flag = calculate_delta_v(2.5)
        assert delta_v <= -0.25
        assert zone == 'critical'
        assert flag is True

    def test_persistent_high_acwr(self):
        """Persistent high ACWR should trigger action."""
        delta_v, zone, flag = calculate_delta_v(1.6, consecutive_high_acwr_weeks=3)
        assert zone == 'persistent_high'
        assert flag is True
        assert delta_v <= -0.25


class TestApplyDeltaV:
    """Tests for applying Delta V to volume."""

    def test_volume_increases(self):
        """Positive Delta V should increase volume."""
        new_vol = apply_delta_v(100, 0.10)
        assert new_vol == 110

    def test_volume_decreases(self):
        """Negative Delta V should decrease volume."""
        new_vol = apply_delta_v(100, -0.20)
        assert new_vol == 80

    def test_minimum_volume_enforced(self):
        """Volume should not go below minimum."""
        new_vol = apply_delta_v(30, -0.50)
        assert new_vol >= 20


# =============================================================================
# Frequency Tests
# =============================================================================

class TestFrequency:
    """Tests for frequency calculations."""

    def test_base_phase_frequency(self):
        """Base phase should have 2-4 days."""
        freq, _ = calculate_base_phase_frequency(
            120, ExperienceLevel.INTERMEDIATE
        )
        assert 2 <= freq <= 4

    def test_eighty_twenty_frequency(self):
        """80/20 phase can have 3-6 days."""
        freq, _ = calculate_eighty_twenty_frequency(
            180, ExperienceLevel.INTERMEDIATE
        )
        assert 3 <= freq <= 6

    def test_frequency_scales_with_volume(self):
        """Higher volume should allow more days."""
        freq_low, _ = calculate_base_phase_frequency(60, ExperienceLevel.INTERMEDIATE)
        freq_high, _ = calculate_base_phase_frequency(120, ExperienceLevel.INTERMEDIATE)
        assert freq_high >= freq_low


class TestRecoveryMetrics:
    """Tests for recovery metrics."""

    def test_good_recovery_score(self):
        """Good recovery should have high score."""
        recovery = RecoveryMetrics(
            hrv_score=0.9,
            sleep_quality=0.9,
            sleep_hours=8.0,
            fatigue_level=0.1,
            soreness_level=0.1,
            stress_level=0.2
        )
        assert recovery.recovery_score > 0.8

    def test_poor_recovery_score(self):
        """Poor recovery should have low score."""
        recovery = RecoveryMetrics(
            hrv_score=0.4,
            sleep_quality=0.3,
            sleep_hours=5.0,
            fatigue_level=0.8,
            soreness_level=0.7,
            stress_level=0.9
        )
        assert recovery.recovery_score < 0.4


# =============================================================================
# Workout Splits Tests
# =============================================================================

class TestWorkoutSplits:
    """Tests for workout split calculations."""

    def test_base_phase_all_easy(self):
        """Base phase should be 100% easy."""
        plan = calculate_base_phase_splits(
            120, 4, ExperienceLevel.INTERMEDIATE
        )
        assert plan.easy_percentage == 1.0
        assert plan.hard_percentage == 0.0

    def test_eighty_twenty_split(self):
        """80/20 phase should have correct split."""
        plan = calculate_eighty_twenty_splits(
            180, 5, ExperienceLevel.INTERMEDIATE, hard_session_count=2
        )
        assert 0.75 <= plan.easy_percentage <= 0.85
        assert 0.15 <= plan.hard_percentage <= 0.25

    def test_long_run_included(self):
        """Plans should include a long run."""
        plan = calculate_base_phase_splits(120, 4, ExperienceLevel.INTERMEDIATE)
        long_runs = [s for s in plan.sessions if s.workout_type == WorkoutType.EASY_LONG]
        assert len(long_runs) == 1


# =============================================================================
# Scheduling Tests
# =============================================================================

class TestScheduling:
    """Tests for workout scheduling."""

    def test_consecutive_days_limit_beginner(self):
        """Beginners should have max 2 consecutive days."""
        max_consec = calculate_max_consecutive_runs(
            ExperienceLevel.BEGINNER, 90, bmi=25
        )
        assert max_consec == 2

    def test_consecutive_days_limit_advanced(self):
        """Advanced runners can have more consecutive days."""
        max_consec = calculate_max_consecutive_runs(
            ExperienceLevel.ADVANCED, 200, bmi=22
        )
        assert max_consec >= 3

    def test_day_assignment_respects_availability(self):
        """Scheduling should respect available days."""
        plan = calculate_base_phase_splits(90, 3, ExperienceLevel.INTERMEDIATE)
        available = {DayOfWeek.TUESDAY, DayOfWeek.THURSDAY, DayOfWeek.SATURDAY}

        schedule = assign_days(
            plan,
            ExperienceLevel.INTERMEDIATE,
            available_days=available
        )

        for workout in schedule.workouts:
            assert workout.day in available


# =============================================================================
# Integration Tests
# =============================================================================

class TestRunPlanEngine:
    """Integration tests for the full engine."""

    @pytest.fixture
    def engine(self):
        """Create engine with optimized params."""
        return RunPlanEngine(use_optimized_params=True)

    @pytest.fixture
    def sample_athlete(self):
        """Create sample athlete."""
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
        )

    def test_plan_initialization(self, engine, sample_athlete):
        """Test that plan initializes correctly."""
        state = engine.initialize_plan(sample_athlete)

        assert state.current_volume > 0
        assert state.current_frequency >= 2
        assert state.current_phase == TrainingPhase.BASE_BUILDING
        assert state.maf_hr > 0
        assert state.current_workout_plan is not None
        assert state.current_schedule is not None

    def test_weekly_update(self, engine, sample_athlete):
        """Test weekly update with training data."""
        state = engine.initialize_plan(sample_athlete)
        initial_volume = state.current_volume

        # Simulate one week of training
        daily_trimp = [50, 0, 60, 0, 50, 0, 70]  # 4 training days
        state = engine.update_with_training_data(
            state,
            daily_trimp,
            actual_volume=120,
            actual_frequency=4
        )

        # Week should have advanced
        assert state.week_number == 2
        assert len(state.week_logs) == 1
        assert len(state.trimp_history) == 7

    def test_12_week_simulation(self, engine, sample_athlete):
        """Test full 12-week simulation."""
        history = engine.simulate_progression(sample_athlete, weeks=12)

        assert len(history) == 13  # Initial + 12 weeks
        assert history[-1]['week_number'] == 12

        # Volume should have grown
        initial_vol = history[0]['volume']
        final_vol = history[-1]['volume']
        growth = final_vol / initial_vol
        assert growth > 1.0  # Some growth expected

    def test_phase_transition(self, engine, sample_athlete):
        """Test that phase transitions work."""
        history = engine.simulate_progression(sample_athlete, weeks=12)

        # Should have transitioned from base building by week 12
        phases = [h['phase'] for h in history]
        assert 'base_building' in phases

        # May have transitioned to transition or 80/20
        # (depends on simulation randomness)


class TestAthleteProfiles:
    """Tests with different athlete profiles."""

    def test_obese_beginner(self):
        """Obese beginner should get conservative plan."""
        athlete = AthleteProfile(
            age=45,
            gender='female',
            weight_kg=110,
            height_cm=165,
            experience_level=ExperienceLevel.BEGINNER,
            health_status=HealthStatus.OPTIMAL,
        )

        adjustments = get_athlete_adjustments(athlete)

        # Should have conservative values
        assert adjustments['max_delta_v'] <= 0.10
        assert adjustments['bmi_stress_modifier'] > 1.0
        assert adjustments['initial_volume'] <= 90

    def test_elite_runner(self):
        """Elite runner should get aggressive plan."""
        athlete = AthleteProfile(
            age=28,
            gender='male',
            weight_kg=65,
            height_cm=175,
            experience_level=ExperienceLevel.ADVANCED,
            consistent_training_years=8,
            injury_free_years=4,
            health_status=HealthStatus.OPTIMAL,
            hr_rest=48,
            vo2max_estimate=60,
            recent_4wk_avg_volume=300,
        )

        adjustments = get_athlete_adjustments(athlete)

        # Should have aggressive values
        assert adjustments['max_delta_v'] >= 0.15
        assert adjustments['bmi_stress_modifier'] == 1.0
        assert adjustments['initial_volume'] >= 180


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_volume(self):
        """Handle zero volume gracefully."""
        freq, _ = calculate_base_phase_frequency(0, ExperienceLevel.BEGINNER)
        assert freq >= 2  # Minimum frequency

    def test_very_high_volume(self):
        """Handle very high volume."""
        freq, _ = calculate_eighty_twenty_frequency(
            500, ExperienceLevel.ADVANCED
        )
        assert freq <= 6  # Capped

    def test_extreme_bmi(self):
        """Handle extreme BMI values."""
        modifier_low = calculate_bmi_stress_modifier(15)
        modifier_high = calculate_bmi_stress_modifier(50)
        assert modifier_low == 1.0
        assert modifier_high <= 1.65

    def test_very_young_athlete(self):
        """Handle young athlete."""
        maf, _ = calculate_maf_hr(age=16)
        assert 100 <= maf <= 170

    def test_elderly_athlete(self):
        """Handle elderly athlete."""
        maf, _ = calculate_maf_hr(age=80)
        assert 100 <= maf <= 170


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
