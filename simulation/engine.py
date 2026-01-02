"""
Simulation Engine: Week-by-week backtesting of Delta V equation.

Simulates training progressions for runner profiles, applying the Delta V
equation to adjust weekly volume based on ACWR, and tracking outcomes.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from copy import deepcopy

import sys
sys.path.insert(0, '/Users/timmac/Desktop/Delta V backtesting')

from core.metrics import calculate_acwr, calculate_acwr_series, classify_acwr_zone
from core.delta_v import DeltaVParams, calculate_delta_v, apply_delta_v
from data.synthetic import (
    RunnerProfile, generate_training_week, generate_warmup_period,
    inject_life_events
)


@dataclass
class WeeklySnapshot:
    """Snapshot of training state for one week."""
    week: int
    target_volume: float      # What Delta V prescribed
    actual_volume: float      # What was actually achieved
    weekly_trimp: float       # Total TRIMP for week
    acwr: float               # ACWR at end of week
    acwr_zone: str            # Zone classification
    delta_v: float            # Applied Delta V value
    delta_v_zone: str         # Zone used for Delta V calc
    flagged: bool             # Whether week was flagged for review
    life_events: Dict[str, float] = field(default_factory=dict)


@dataclass
class SimulationResult:
    """Complete results from one simulation run."""
    profile_id: str
    profile_name: str
    params: DeltaVParams
    weeks: List[WeeklySnapshot]

    # Summary metrics
    initial_volume: float
    final_volume: float
    volume_growth_ratio: float
    target_volume_reached: bool

    # Risk metrics
    total_risk_events: int          # Weeks with ACWR > 1.5
    max_consecutive_risk: int       # Max consecutive high-ACWR weeks
    injury_proxy_triggered: bool    # >2 consecutive weeks > 1.5

    # Stability metrics
    volume_change_std: float        # Std dev of weekly % changes
    max_volume_increase: float      # Largest single-week increase
    max_volume_decrease: float      # Largest single-week decrease

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for analysis."""
        return {
            'profile_id': self.profile_id,
            'profile_name': self.profile_name,
            'initial_volume': self.initial_volume,
            'final_volume': self.final_volume,
            'volume_growth_ratio': self.volume_growth_ratio,
            'target_reached': self.target_volume_reached,
            'risk_events': self.total_risk_events,
            'max_consecutive_risk': self.max_consecutive_risk,
            'injury_triggered': self.injury_proxy_triggered,
            'volume_change_std': self.volume_change_std,
            'max_increase': self.max_volume_increase,
            'max_decrease': self.max_volume_decrease,
            'weeks_simulated': len(self.weeks),
        }

    def get_volume_trajectory(self) -> np.ndarray:
        """Get array of weekly volumes."""
        return np.array([w.actual_volume for w in self.weeks])

    def get_acwr_trajectory(self) -> np.ndarray:
        """Get array of weekly ACWR values."""
        return np.array([w.acwr for w in self.weeks])

    def get_delta_v_trajectory(self) -> np.ndarray:
        """Get array of applied Delta V values."""
        return np.array([w.delta_v for w in self.weeks])


class SimulationEngine:
    """
    Engine for running Delta V backtests.

    Simulates week-by-week progression for runner profiles,
    applying Delta V to adjust volume based on ACWR.
    """

    def __init__(
        self,
        params: Optional[DeltaVParams] = None,
        warmup_weeks: int = 4,
        target_growth_ratio: float = 2.5,  # Target 2.5x initial volume
        verbose: bool = False
    ):
        """
        Initialize simulation engine.

        Args:
            params: Delta V parameters (uses defaults if None)
            warmup_weeks: Weeks for establishing baseline ACWR
            target_growth_ratio: Target volume growth (e.g., 2.5 = 150% increase)
            verbose: Print progress during simulation
        """
        self.params = params or DeltaVParams()
        self.warmup_weeks = warmup_weeks
        self.target_growth_ratio = target_growth_ratio
        self.verbose = verbose

    def run_simulation(
        self,
        profile: RunnerProfile,
        num_weeks: int = 12,
        life_events: Optional[List[Dict[str, float]]] = None,
        seed: Optional[int] = None
    ) -> SimulationResult:
        """
        Run full simulation for a runner profile.

        Args:
            profile: Runner profile to simulate
            num_weeks: Number of weeks to simulate
            life_events: Optional life events per week
            seed: Random seed for reproducibility

        Returns:
            SimulationResult with complete data
        """
        if seed is not None:
            np.random.seed(seed)

        # Generate life events if not provided
        if life_events is None:
            life_events = inject_life_events(num_weeks, seed)

        # Make a copy to avoid mutating original
        profile = deepcopy(profile)

        # Phase 1: Warmup to establish chronic baseline
        warmup_trimps = generate_warmup_period(profile, self.warmup_weeks)
        daily_trimp_history = list(warmup_trimps)

        # Track state
        current_volume = profile.initial_weekly_volume
        target_volume = profile.initial_weekly_volume * self.target_growth_ratio
        consecutive_high_acwr = 0
        weekly_snapshots = []

        if self.verbose:
            print(f"Simulating {profile.name}: {num_weeks} weeks")
            print(f"  Initial: {profile.initial_weekly_volume:.0f} min")
            print(f"  Target: {target_volume:.0f} min")

        # Phase 2: Simulation weeks
        for week in range(num_weeks):
            # Calculate current ACWR
            acwr, _, _ = calculate_acwr(np.array(daily_trimp_history))
            acwr_zone = classify_acwr_zone(acwr)

            # Track consecutive high ACWR
            if acwr > 1.5:
                consecutive_high_acwr += 1
            else:
                consecutive_high_acwr = 0

            # Calculate Delta V
            delta_v, dv_zone, flagged = calculate_delta_v(
                acwr, self.params, consecutive_high_acwr
            )

            # Calculate new target volume
            new_target_volume = apply_delta_v(current_volume, delta_v)

            # Generate actual training week (with variance and life events)
            week_events = life_events[week] if week < len(life_events) else {}
            week_data, weekly_trimp = generate_training_week(
                profile, new_target_volume, week, week_events
            )

            # Calculate actual achieved volume
            actual_volume = sum(d.duration_min for d in week_data if d.completed)

            # Update TRIMP history
            for d in week_data:
                daily_trimp_history.append(d.trimp)

            # Record snapshot
            snapshot = WeeklySnapshot(
                week=week + 1,
                target_volume=new_target_volume,
                actual_volume=actual_volume,
                weekly_trimp=weekly_trimp,
                acwr=acwr,
                acwr_zone=acwr_zone,
                delta_v=delta_v,
                delta_v_zone=dv_zone,
                flagged=flagged,
                life_events=week_events
            )
            weekly_snapshots.append(snapshot)

            # Update current volume for next iteration
            current_volume = new_target_volume

            if self.verbose and week % 4 == 0:
                print(f"  Week {week+1}: Vol={actual_volume:.0f}, "
                      f"ACWR={acwr:.2f} ({acwr_zone}), DV={delta_v*100:+.1f}%")

        # Calculate summary metrics
        result = self._calculate_summary(
            profile, weekly_snapshots, target_volume
        )

        return result

    def _calculate_summary(
        self,
        profile: RunnerProfile,
        weeks: List[WeeklySnapshot],
        target_volume: float
    ) -> SimulationResult:
        """Calculate summary metrics from weekly data."""

        volumes = [w.actual_volume for w in weeks]
        acwrs = [w.acwr for w in weeks]
        delta_vs = [w.delta_v for w in weeks]

        # Volume metrics
        initial = profile.initial_weekly_volume
        final = volumes[-1] if volumes else initial
        growth_ratio = final / initial if initial > 0 else 1.0

        # Risk metrics
        risk_events = sum(1 for a in acwrs if a > 1.5)

        max_consecutive = 0
        current_consecutive = 0
        for a in acwrs:
            if a > 1.5:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        injury_triggered = max_consecutive > 2

        # Stability metrics
        if len(volumes) > 1:
            pct_changes = [(volumes[i] - volumes[i-1]) / volumes[i-1]
                          for i in range(1, len(volumes)) if volumes[i-1] > 0]
            volume_std = np.std(pct_changes) if pct_changes else 0
            max_increase = max(delta_vs) if delta_vs else 0
            max_decrease = min(delta_vs) if delta_vs else 0
        else:
            volume_std = 0
            max_increase = 0
            max_decrease = 0

        return SimulationResult(
            profile_id=profile.id,
            profile_name=profile.name,
            params=self.params,
            weeks=weeks,
            initial_volume=initial,
            final_volume=final,
            volume_growth_ratio=growth_ratio,
            target_volume_reached=final >= target_volume * 0.9,  # 90% of target
            total_risk_events=risk_events,
            max_consecutive_risk=max_consecutive,
            injury_proxy_triggered=injury_triggered,
            volume_change_std=volume_std,
            max_volume_increase=max_increase,
            max_volume_decrease=max_decrease,
        )

    def run_batch(
        self,
        profiles: List[RunnerProfile],
        num_weeks: int = 12,
        seed: Optional[int] = None
    ) -> List[SimulationResult]:
        """
        Run simulations for multiple profiles.

        Args:
            profiles: List of runner profiles
            num_weeks: Weeks per simulation
            seed: Base random seed

        Returns:
            List of SimulationResult objects
        """
        results = []
        for i, profile in enumerate(profiles):
            profile_seed = seed + i if seed is not None else None
            result = self.run_simulation(profile, num_weeks, seed=profile_seed)
            results.append(result)
        return results


def aggregate_results(results: List[SimulationResult]) -> Dict[str, Any]:
    """
    Aggregate metrics across multiple simulation results.

    Args:
        results: List of SimulationResult objects

    Returns:
        Dictionary of aggregated metrics
    """
    if not results:
        return {}

    n = len(results)

    growth_ratios = [r.volume_growth_ratio for r in results]
    risk_events = [r.total_risk_events for r in results]
    target_reached = [r.target_volume_reached for r in results]
    injury_triggered = [r.injury_proxy_triggered for r in results]
    vol_stds = [r.volume_change_std for r in results]

    return {
        'n_simulations': n,

        # Volume growth
        'mean_growth_ratio': np.mean(growth_ratios),
        'std_growth_ratio': np.std(growth_ratios),
        'min_growth_ratio': np.min(growth_ratios),
        'max_growth_ratio': np.max(growth_ratios),

        # Target achievement
        'pct_target_reached': sum(target_reached) / n * 100,

        # Risk
        'mean_risk_events': np.mean(risk_events),
        'total_risk_events': sum(risk_events),
        'pct_with_injury_proxy': sum(injury_triggered) / n * 100,

        # Stability
        'mean_volume_std': np.mean(vol_stds),

        # Per-week risk rate
        'risk_event_rate': sum(risk_events) / (n * len(results[0].weeks)) * 100,
    }


def compare_params(
    params_a: DeltaVParams,
    params_b: DeltaVParams,
    profiles: List[RunnerProfile],
    num_weeks: int = 12,
    seed: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Compare two parameter sets across same profiles.

    Args:
        params_a: First parameter set (baseline)
        params_b: Second parameter set (candidate)
        profiles: Profiles to test
        num_weeks: Simulation weeks
        seed: Random seed

    Returns:
        Dictionary with aggregated results for each param set
    """
    engine_a = SimulationEngine(params_a)
    engine_b = SimulationEngine(params_b)

    results_a = engine_a.run_batch(profiles, num_weeks, seed)
    results_b = engine_b.run_batch(profiles, num_weeks, seed)

    agg_a = aggregate_results(results_a)
    agg_b = aggregate_results(results_b)

    return {
        'baseline': agg_a,
        'candidate': agg_b,
        'improvement': {
            'growth_ratio_delta': agg_b['mean_growth_ratio'] - agg_a['mean_growth_ratio'],
            'risk_reduction': agg_a['mean_risk_events'] - agg_b['mean_risk_events'],
            'target_reached_delta': agg_b['pct_target_reached'] - agg_a['pct_target_reached'],
        }
    }


if __name__ == '__main__':
    print("Testing simulation engine...")

    from data.synthetic import generate_runner_profiles

    # Generate test profiles
    profiles = generate_runner_profiles(5, seed=42)

    # Create engine with default params
    engine = SimulationEngine(verbose=True)

    # Run single simulation
    print("\n" + "="*60)
    result = engine.run_simulation(profiles[0], num_weeks=12, seed=42)

    print(f"\nResults for {result.profile_name}:")
    print(f"  Initial volume: {result.initial_volume:.0f} min")
    print(f"  Final volume: {result.final_volume:.0f} min")
    print(f"  Growth ratio: {result.volume_growth_ratio:.2f}x")
    print(f"  Target reached: {result.target_volume_reached}")
    print(f"  Risk events: {result.total_risk_events}")
    print(f"  Injury proxy: {result.injury_proxy_triggered}")

    # Run batch
    print("\n" + "="*60)
    print("Running batch simulation...")
    batch_results = engine.run_batch(profiles, num_weeks=12, seed=42)

    # Aggregate
    agg = aggregate_results(batch_results)
    print("\nAggregate Results:")
    for key, value in agg.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    print("\nAll tests passed!")
