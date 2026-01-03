"""
Delta V Parameter Optimization
===============================

Multi-objective optimization to find optimal Delta V parameters that:
1. MAXIMIZE volume growth over time
2. MINIMIZE injury risk (stay in safe ACWR zones)

Uses empirical injury risk rates from validation study:
- Low (ACWR < 0.8): 1.44% daily injury rate, RR = 1.21
- Optimal (0.8-1.3): 1.19% daily injury rate, RR = 1.00 (reference)
- Caution (1.3-1.5): 1.39% daily injury rate, RR = 1.17
- High (1.5-2.0): 1.61% daily injury rate, RR = 1.36
- Critical (≥ 2.0): 1.73% daily injury rate, RR = 1.46

Author: Claude (AI Research Assistant)
Date: 2026-01-02
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy.optimize import differential_evolution, minimize
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
import json
from datetime import datetime

from core.delta_v import DeltaVParams, calculate_delta_v, PARAM_BOUNDS


# ═══════════════════════════════════════════════════════════════════════════════
# EMPIRICAL INJURY RISK MODEL (from validation study)
# ═══════════════════════════════════════════════════════════════════════════════

# Daily injury probability by ACWR zone (from Experiment 003)
INJURY_RATES = {
    'low': 0.0144,        # 1.44% per day
    'optimal': 0.0119,    # 1.19% per day (lowest)
    'caution': 0.0139,    # 1.39% per day
    'red': 0.0161,        # 1.61% per day
    'critical': 0.0173,   # 1.73% per day
}

# Relative risk vs optimal zone
RELATIVE_RISK = {
    'low': 1.21,
    'optimal': 1.00,
    'caution': 1.17,
    'red': 1.36,
    'critical': 1.46,
}


def get_injury_probability(acwr: float, params: DeltaVParams = None) -> float:
    """
    Get daily injury probability based on ACWR.
    Uses linear interpolation within zones for smooth gradient.
    """
    if params is None:
        params = DeltaVParams()

    # Zone boundaries
    t_low = params.threshold_low
    t_opt_high = params.threshold_optimal_high
    t_caution = params.threshold_caution
    t_critical = params.threshold_critical

    # Get base rates
    if acwr >= t_critical:
        return INJURY_RATES['critical']
    elif acwr >= t_caution:
        # Linear interpolation between red and critical
        alpha = (acwr - t_caution) / (t_critical - t_caution)
        return INJURY_RATES['red'] + alpha * (INJURY_RATES['critical'] - INJURY_RATES['red'])
    elif acwr >= t_opt_high:
        # Linear interpolation between caution and red
        alpha = (acwr - t_opt_high) / (t_caution - t_opt_high)
        return INJURY_RATES['caution'] + alpha * (INJURY_RATES['red'] - INJURY_RATES['caution'])
    elif acwr >= t_low:
        # Optimal zone - lowest risk
        return INJURY_RATES['optimal']
    else:
        # Low zone - slightly elevated
        return INJURY_RATES['low']


# ═══════════════════════════════════════════════════════════════════════════════
# ATHLETE SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimulationResult:
    """Results from a single athlete simulation."""
    initial_volume: float
    final_volume: float
    volume_growth: float
    growth_rate: float  # Multiplicative growth (final/initial)
    days_simulated: int
    injury_occurred: bool
    injury_day: Optional[int]
    days_in_zones: Dict[str, int]
    mean_acwr: float
    max_acwr: float
    total_injury_exposure: float  # Sum of daily injury probabilities


def simulate_athlete(
    params: DeltaVParams,
    initial_volume: float = 30.0,  # km/week
    simulation_days: int = 84,     # 12 weeks
    chronic_window: int = 28,
    acute_window: int = 7,
    random_seed: int = None
) -> SimulationResult:
    """
    Simulate an athlete following Delta V recommendations.

    Args:
        params: Delta V parameters to test
        initial_volume: Starting weekly volume (km)
        simulation_days: Number of days to simulate
        chronic_window: Days for chronic load calculation
        acute_window: Days for acute load calculation
        random_seed: For reproducibility

    Returns:
        SimulationResult with outcomes
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Initialize daily loads (assume 5 training days per week initially)
    daily_volume = initial_volume / 5
    load_history = [daily_volume] * chronic_window  # Pre-fill with baseline

    # Track outcomes
    days_in_zones = {'low': 0, 'optimal': 0, 'caution': 0, 'red': 0, 'critical': 0}
    acwr_history = []
    injury_occurred = False
    injury_day = None
    total_injury_exposure = 0.0

    consecutive_high_weeks = 0
    weekly_high_acwr_days = 0

    for day in range(simulation_days):
        # Calculate current ACWR
        acute_load = np.mean(load_history[-acute_window:])
        chronic_load = np.mean(load_history[-chronic_window:])
        acwr = acute_load / chronic_load if chronic_load > 0.1 else 1.0

        acwr_history.append(acwr)

        # Classify zone
        if acwr >= params.threshold_critical:
            zone = 'critical'
        elif acwr >= params.threshold_caution:
            zone = 'red'
        elif acwr >= params.threshold_optimal_high:
            zone = 'caution'
        elif acwr >= params.threshold_low:
            zone = 'optimal'
        else:
            zone = 'low'

        days_in_zones[zone] += 1

        # Track high ACWR for persistence
        if acwr > params.threshold_caution:
            weekly_high_acwr_days += 1
        if day % 7 == 6:  # End of week
            if weekly_high_acwr_days >= 3:  # 3+ days above threshold
                consecutive_high_weeks += 1
            else:
                consecutive_high_weeks = 0
            weekly_high_acwr_days = 0

        # Check for injury (probabilistic)
        injury_prob = get_injury_probability(acwr, params)
        total_injury_exposure += injury_prob

        if np.random.random() < injury_prob:
            injury_occurred = True
            injury_day = day
            break  # Simulation ends on injury

        # Get Delta V recommendation
        delta_v, _, _ = calculate_delta_v(acwr, params, consecutive_high_weeks)

        # Apply recommendation to weekly volume
        # (We adjust daily volume based on weekly recommendation)
        current_weekly_vol = sum(load_history[-7:])
        new_weekly_vol = current_weekly_vol * (1 + delta_v / 7)  # Gradual daily change

        # Add today's training (assuming 5 training days per week pattern)
        if day % 7 < 5:  # Training day
            new_daily = new_weekly_vol / 5
            # Add some natural variation (±10%)
            new_daily *= (1 + np.random.uniform(-0.1, 0.1))
            load_history.append(max(0.1, new_daily))
        else:  # Rest day
            load_history.append(0.1)  # Minimal activity

    # Calculate final metrics
    final_weekly_volume = sum(load_history[-7:]) if len(load_history) >= 7 else initial_volume
    volume_growth = final_weekly_volume - initial_volume
    growth_rate = final_weekly_volume / initial_volume

    return SimulationResult(
        initial_volume=initial_volume,
        final_volume=final_weekly_volume,
        volume_growth=volume_growth,
        growth_rate=growth_rate,
        days_simulated=day + 1 if injury_occurred else simulation_days,
        injury_occurred=injury_occurred,
        injury_day=injury_day,
        days_in_zones=days_in_zones,
        mean_acwr=np.mean(acwr_history) if acwr_history else 1.0,
        max_acwr=np.max(acwr_history) if acwr_history else 1.0,
        total_injury_exposure=total_injury_exposure
    )


def simulate_population(
    params: DeltaVParams,
    n_athletes: int = 100,
    simulation_days: int = 84,
    initial_volume_range: Tuple[float, float] = (20.0, 50.0),
    base_seed: int = 42
) -> Dict:
    """
    Simulate a population of athletes.

    Returns:
        Dict with population-level metrics
    """
    results = []

    for i in range(n_athletes):
        initial_vol = np.random.uniform(*initial_volume_range)
        result = simulate_athlete(
            params,
            initial_volume=initial_vol,
            simulation_days=simulation_days,
            random_seed=base_seed + i
        )
        results.append(result)

    # Aggregate metrics
    injured = sum(1 for r in results if r.injury_occurred)
    completed = sum(1 for r in results if not r.injury_occurred)

    growth_rates = [r.growth_rate for r in results if not r.injury_occurred]
    mean_growth = np.mean(growth_rates) if growth_rates else 0

    all_growth_rates = [r.growth_rate for r in results]

    # Zone time distribution
    total_days = sum(r.days_simulated for r in results)
    zone_pct = {}
    for zone in ['low', 'optimal', 'caution', 'red', 'critical']:
        zone_days = sum(r.days_in_zones[zone] for r in results)
        zone_pct[zone] = zone_days / total_days * 100 if total_days > 0 else 0

    return {
        'n_athletes': n_athletes,
        'injury_rate': injured / n_athletes,
        'completion_rate': completed / n_athletes,
        'mean_growth_rate_healthy': mean_growth,
        'mean_growth_rate_all': np.mean(all_growth_rates),
        'median_growth_rate': np.median(all_growth_rates),
        'pct_achieved_2x': sum(1 for r in all_growth_rates if r >= 2.0) / n_athletes * 100,
        'pct_achieved_1_5x': sum(1 for r in all_growth_rates if r >= 1.5) / n_athletes * 100,
        'zone_distribution': zone_pct,
        'mean_injury_exposure': np.mean([r.total_injury_exposure for r in results]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMIZATION OBJECTIVE
# ═══════════════════════════════════════════════════════════════════════════════

def params_from_vector(x: np.ndarray, param_names: List[str]) -> DeltaVParams:
    """Convert optimization vector to DeltaVParams."""
    param_dict = {}
    for i, name in enumerate(param_names):
        param_dict[name] = x[i]

    # Fill in defaults for non-optimized params
    defaults = DeltaVParams()
    for field in ['threshold_low', 'threshold_optimal_high', 'threshold_caution',
                  'threshold_critical', 'green_base', 'green_min', 'green_max',
                  'low_base', 'low_min', 'low_max', 'caution_value',
                  'red_base', 'red_min', 'red_max', 'critical_value']:
        if field not in param_dict:
            param_dict[field] = getattr(defaults, field)

    return DeltaVParams(**param_dict)


def objective_function(
    x: np.ndarray,
    param_names: List[str],
    n_athletes: int = 50,
    simulation_days: int = 84,
    injury_weight: float = 10.0,
    target_growth: float = 2.0,
    base_seed: int = 42
) -> float:
    """
    Optimization objective: Maximize growth while minimizing injury.

    Objective = -(mean_growth - injury_weight * injury_rate)

    We negate because scipy minimizes.

    Args:
        x: Parameter vector
        param_names: Names of parameters being optimized
        n_athletes: Number of athletes to simulate
        simulation_days: Simulation duration
        injury_weight: Penalty multiplier for injuries
        target_growth: Target growth rate (for bonus)
        base_seed: Random seed for reproducibility
    """
    try:
        params = params_from_vector(x, param_names)

        # Validate parameters
        valid, _ = params.validate()
        if not valid:
            return 1e6  # Invalid parameters

        # Run simulation
        results = simulate_population(
            params,
            n_athletes=n_athletes,
            simulation_days=simulation_days,
            base_seed=base_seed
        )

        # Calculate objective
        growth = results['mean_growth_rate_all']
        injury_rate = results['injury_rate']

        # Objective: maximize growth, minimize injury
        # Penalty for injuries, bonus for hitting target
        objective = growth - injury_weight * injury_rate

        # Bonus for staying in optimal zone
        optimal_pct = results['zone_distribution'].get('optimal', 0)
        objective += 0.01 * optimal_pct  # Small bonus

        return -objective  # Negate for minimization

    except Exception as e:
        print(f"Error in objective: {e}")
        return 1e6


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════════

class DeltaVOptimizer:
    """
    Optimizer for Delta V parameters using differential evolution.
    """

    def __init__(
        self,
        params_to_optimize: List[str] = None,
        n_athletes: int = 50,
        simulation_days: int = 84,
        injury_weight: float = 10.0
    ):
        """
        Initialize optimizer.

        Args:
            params_to_optimize: List of parameter names to optimize.
                               If None, optimizes the main magnitude parameters.
            n_athletes: Athletes per simulation
            simulation_days: Days to simulate
            injury_weight: Penalty weight for injuries
        """
        if params_to_optimize is None:
            # Default: optimize the magnitude parameters, keep thresholds fixed
            params_to_optimize = [
                'green_base', 'green_min', 'green_max',
                'low_base', 'low_min', 'low_max',
                'caution_value',
                'red_base', 'red_min', 'red_max',
                'critical_value'
            ]

        self.params_to_optimize = params_to_optimize
        self.n_athletes = n_athletes
        self.simulation_days = simulation_days
        self.injury_weight = injury_weight

        # Get bounds for selected parameters
        self.bounds = [PARAM_BOUNDS[p] for p in params_to_optimize]

        self.best_params = None
        self.best_score = None
        self.history = []

    def optimize(
        self,
        maxiter: int = 50,
        popsize: int = 15,
        seed: int = 42,
        verbose: bool = True
    ) -> DeltaVParams:
        """
        Run optimization.

        Args:
            maxiter: Maximum iterations
            popsize: Population size for differential evolution
            seed: Random seed
            verbose: Print progress

        Returns:
            Optimized DeltaVParams
        """
        if verbose:
            print("=" * 70)
            print("DELTA V PARAMETER OPTIMIZATION")
            print("=" * 70)
            print(f"Optimizing: {self.params_to_optimize}")
            print(f"Athletes per sim: {self.n_athletes}")
            print(f"Simulation days: {self.simulation_days}")
            print(f"Injury weight: {self.injury_weight}")
            print()

        iteration_count = [0]  # Use list to allow modification in closure
        best_so_far = [float('inf')]

        def callback(xk, convergence):
            """Callback to track progress."""
            iteration_count[0] += 1
            # Calculate current objective
            current_obj = objective_function(
                xk, self.params_to_optimize, self.n_athletes,
                self.simulation_days, self.injury_weight, 2.0, seed
            )
            if current_obj < best_so_far[0]:
                best_so_far[0] = current_obj

            self.history.append({
                'iteration': iteration_count[0],
                'x': xk.tolist(),
                'objective': -current_obj,
                'convergence': convergence
            })
            if verbose and iteration_count[0] % 5 == 0:
                print(f"  Iteration {iteration_count[0]}: current = {-current_obj:.4f}, best = {-best_so_far[0]:.4f}")

        # Run differential evolution
        result = differential_evolution(
            objective_function,
            bounds=self.bounds,
            args=(
                self.params_to_optimize,
                self.n_athletes,
                self.simulation_days,
                self.injury_weight,
                2.0,  # target_growth
                seed
            ),
            maxiter=maxiter,
            popsize=popsize,
            seed=seed,
            callback=callback,
            disp=verbose,
            workers=1,  # Single-threaded for reproducibility
            updating='deferred',
            polish=True
        )

        self.best_score = -result.fun
        self.best_params = params_from_vector(result.x, self.params_to_optimize)

        if verbose:
            print()
            print("=" * 70)
            print("OPTIMIZATION COMPLETE")
            print("=" * 70)
            print(f"Best objective: {self.best_score:.4f}")
            print()
            print("Optimized Parameters:")
            for i, name in enumerate(self.params_to_optimize):
                print(f"  {name}: {result.x[i]:.4f}")

        return self.best_params

    def evaluate(self, params: DeltaVParams, n_trials: int = 5, verbose: bool = True) -> Dict:
        """
        Evaluate parameters with multiple trials for robustness.
        """
        results = []
        for trial in range(n_trials):
            r = simulate_population(
                params,
                n_athletes=100,
                simulation_days=self.simulation_days,
                base_seed=42 + trial * 1000
            )
            results.append(r)

        # Average across trials
        avg_results = {
            'mean_growth_rate': np.mean([r['mean_growth_rate_all'] for r in results]),
            'std_growth_rate': np.std([r['mean_growth_rate_all'] for r in results]),
            'mean_injury_rate': np.mean([r['injury_rate'] for r in results]),
            'std_injury_rate': np.std([r['injury_rate'] for r in results]),
            'mean_completion_rate': np.mean([r['completion_rate'] for r in results]),
            'pct_achieved_2x': np.mean([r['pct_achieved_2x'] for r in results]),
            'pct_achieved_1_5x': np.mean([r['pct_achieved_1_5x'] for r in results]),
        }

        if verbose:
            print("\nEVALUATION RESULTS (5 trials, 100 athletes each):")
            print("-" * 50)
            print(f"  Growth rate: {avg_results['mean_growth_rate']:.2f}x ± {avg_results['std_growth_rate']:.2f}")
            print(f"  Injury rate: {avg_results['mean_injury_rate']*100:.1f}% ± {avg_results['std_injury_rate']*100:.1f}%")
            print(f"  Completion rate: {avg_results['mean_completion_rate']*100:.1f}%")
            print(f"  Achieved 1.5x: {avg_results['pct_achieved_1_5x']:.1f}%")
            print(f"  Achieved 2.0x: {avg_results['pct_achieved_2x']:.1f}%")

        return avg_results


def run_optimization():
    """Run the full optimization pipeline."""
    print("=" * 70)
    print("DELTA V OPTIMIZATION EXPERIMENT")
    print("=" * 70)
    print()

    # 1. Evaluate default parameters
    print("-" * 70)
    print("STEP 1: BASELINE (DEFAULT PARAMETERS)")
    print("-" * 70)

    default_params = DeltaVParams()
    optimizer = DeltaVOptimizer(
        n_athletes=50,
        simulation_days=84,
        injury_weight=10.0
    )

    print("\nDefault parameter values:")
    for name in optimizer.params_to_optimize:
        print(f"  {name}: {getattr(default_params, name)}")

    baseline = optimizer.evaluate(default_params, n_trials=5)

    # 2. Run optimization
    print("\n" + "-" * 70)
    print("STEP 2: OPTIMIZATION")
    print("-" * 70)

    optimized_params = optimizer.optimize(
        maxiter=20,
        popsize=8,
        seed=42,
        verbose=True
    )

    # 3. Evaluate optimized parameters
    print("\n" + "-" * 70)
    print("STEP 3: EVALUATE OPTIMIZED PARAMETERS")
    print("-" * 70)

    optimized_results = optimizer.evaluate(optimized_params, n_trials=5)

    # 4. Compare
    print("\n" + "=" * 70)
    print("COMPARISON: DEFAULT vs OPTIMIZED")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Default':>15} {'Optimized':>15} {'Improvement':>15}")
    print("-" * 70)

    growth_imp = (optimized_results['mean_growth_rate'] / baseline['mean_growth_rate'] - 1) * 100
    injury_imp = (baseline['mean_injury_rate'] - optimized_results['mean_injury_rate']) / baseline['mean_injury_rate'] * 100

    print(f"{'Growth Rate':<25} {baseline['mean_growth_rate']:>15.2f}x {optimized_results['mean_growth_rate']:>15.2f}x {growth_imp:>+14.1f}%")
    print(f"{'Injury Rate':<25} {baseline['mean_injury_rate']*100:>14.1f}% {optimized_results['mean_injury_rate']*100:>14.1f}% {injury_imp:>+14.1f}%")
    print(f"{'Achieved 2x Target':<25} {baseline['pct_achieved_2x']:>14.1f}% {optimized_results['pct_achieved_2x']:>14.1f}%")

    # 5. Save results
    print("\n" + "-" * 70)
    print("STEP 4: SAVING RESULTS")
    print("-" * 70)

    output_dir = Path(__file__).parent.parent / "experiments" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
        'timestamp': timestamp,
        'baseline': baseline,
        'optimized': optimized_results,
        'optimized_params': optimized_params.to_dict(),
        'improvement': {
            'growth_rate_pct': growth_imp,
            'injury_rate_reduction_pct': injury_imp
        }
    }

    output_file = output_dir / f"optimization_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")

    # 6. Print final optimized equation
    print("\n" + "=" * 70)
    print("OPTIMIZED DELTA V EQUATION")
    print("=" * 70)

    print("\nOptimized Parameters:")
    print("-" * 40)
    for name in ['green_base', 'green_min', 'green_max',
                 'low_base', 'low_min', 'low_max',
                 'caution_value',
                 'red_base', 'red_min', 'red_max',
                 'critical_value']:
        default_val = getattr(default_params, name)
        opt_val = getattr(optimized_params, name)
        change = (opt_val - default_val) / abs(default_val) * 100 if default_val != 0 else 0
        print(f"  {name:<20} {default_val:>8.3f} -> {opt_val:>8.3f} ({change:>+6.1f}%)")

    # Save optimized params
    params_file = Path(__file__).parent.parent / "optimized_params_v2.json"
    with open(params_file, 'w') as f:
        json.dump(optimized_params.to_dict(), f, indent=2)
    print(f"\nOptimized parameters saved to: {params_file}")

    return optimizer, optimized_params


if __name__ == "__main__":
    optimizer, params = run_optimization()
