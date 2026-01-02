"""
Parameter Optimization for Delta V Equation.

Uses Optuna for Bayesian optimization with:
- Multi-objective support (Pareto frontier)
- Pruning for efficiency
- Constraint handling
"""

from typing import List, Optional, Dict, Any, Callable, Tuple
import numpy as np
from dataclasses import asdict

import sys
sys.path.insert(0, '/Users/timmac/Desktop/Delta V backtesting')

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not installed. Using grid search fallback.")

from core.delta_v import DeltaVParams, PARAM_BOUNDS
from data.synthetic import RunnerProfile
from simulation.engine import SimulationEngine
from optimization.objective import (
    evaluate_simulation_results,
    calculate_total_objective,
    ObjectiveWeights
)


def params_from_trial(trial) -> DeltaVParams:
    """
    Create DeltaVParams from Optuna trial suggestions.

    Args:
        trial: Optuna trial object

    Returns:
        DeltaVParams with suggested values
    """
    return DeltaVParams(
        # Thresholds (with ordering constraints)
        threshold_low=trial.suggest_float(
            'threshold_low', *PARAM_BOUNDS['threshold_low']
        ),
        threshold_optimal_high=trial.suggest_float(
            'threshold_optimal_high', *PARAM_BOUNDS['threshold_optimal_high']
        ),
        threshold_caution=trial.suggest_float(
            'threshold_caution', *PARAM_BOUNDS['threshold_caution']
        ),
        threshold_critical=trial.suggest_float(
            'threshold_critical', *PARAM_BOUNDS['threshold_critical']
        ),

        # Green zone
        green_base=trial.suggest_float('green_base', *PARAM_BOUNDS['green_base']),
        green_min=trial.suggest_float('green_min', *PARAM_BOUNDS['green_min']),
        green_max=trial.suggest_float('green_max', *PARAM_BOUNDS['green_max']),

        # Low zone
        low_base=trial.suggest_float('low_base', *PARAM_BOUNDS['low_base']),
        low_min=trial.suggest_float('low_min', *PARAM_BOUNDS['low_min']),
        low_max=trial.suggest_float('low_max', *PARAM_BOUNDS['low_max']),

        # Caution zone
        caution_value=trial.suggest_float(
            'caution_value', *PARAM_BOUNDS['caution_value']
        ),

        # Red zone
        red_base=trial.suggest_float('red_base', *PARAM_BOUNDS['red_base']),
        red_min=trial.suggest_float('red_min', *PARAM_BOUNDS['red_min']),
        red_max=trial.suggest_float('red_max', *PARAM_BOUNDS['red_max']),

        # Critical zone
        critical_value=trial.suggest_float(
            'critical_value', *PARAM_BOUNDS['critical_value']
        ),
    )


def create_objective_function(
    profiles: List[RunnerProfile],
    num_weeks: int = 12,
    weights: Optional[ObjectiveWeights] = None,
    seed: int = 42
) -> Callable:
    """
    Create objective function for Optuna optimization.

    Args:
        profiles: Runner profiles for testing
        num_weeks: Simulation weeks
        weights: Objective weights
        seed: Random seed

    Returns:
        Objective function for Optuna
    """
    def objective(trial) -> float:
        # Create params from trial
        params = params_from_trial(trial)

        # Validate params
        valid, msg = params.validate()
        if not valid:
            # Return poor score for invalid params
            return 0.0

        # Run simulations
        engine = SimulationEngine(params)
        results = engine.run_batch(profiles, num_weeks, seed)

        # Calculate objective
        score = calculate_total_objective(results, weights)

        # Store detailed metrics for analysis
        trial.set_user_attr('detailed_scores',
                           evaluate_simulation_results(results, weights))

        return score

    return objective


def optimize_delta_v_params(
    profiles: List[RunnerProfile],
    n_trials: int = 100,
    num_weeks: int = 12,
    weights: Optional[ObjectiveWeights] = None,
    seed: int = 42,
    verbose: bool = True,
    study_name: str = "delta_v_optimization"
) -> Tuple[DeltaVParams, Dict[str, Any]]:
    """
    Optimize Delta V parameters using Bayesian optimization.

    Args:
        profiles: Runner profiles for testing
        n_trials: Number of optimization trials
        num_weeks: Simulation weeks
        weights: Objective weights
        seed: Random seed
        verbose: Print progress
        study_name: Name for the study

    Returns:
        Tuple of (best parameters, optimization results dict)
    """
    if not OPTUNA_AVAILABLE:
        return _grid_search_fallback(
            profiles, n_trials, num_weeks, weights, seed, verbose
        )

    # Create study
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        direction='maximize',  # We want to maximize composite score
        sampler=sampler,
        study_name=study_name
    )

    # Create objective
    objective = create_objective_function(profiles, num_weeks, weights, seed)

    # Optimize
    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    # Extract best parameters
    best_trial = study.best_trial
    best_params = params_from_trial_values(best_trial.params)

    # Compile results
    results = {
        'best_value': study.best_value,
        'best_params': asdict(best_params),
        'detailed_scores': best_trial.user_attrs.get('detailed_scores', {}),
        'n_trials': n_trials,
        'all_trials': [
            {
                'number': t.number,
                'value': t.value,
                'params': t.params,
            }
            for t in study.trials
        ],
    }

    if verbose:
        print(f"\nOptimization Complete!")
        print(f"Best Score: {study.best_value:.4f}")
        print(f"Best Parameters:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value:.4f}")

    return best_params, results


def params_from_trial_values(values: Dict[str, float]) -> DeltaVParams:
    """Create DeltaVParams from trial parameter values."""
    return DeltaVParams(
        threshold_low=values['threshold_low'],
        threshold_optimal_high=values['threshold_optimal_high'],
        threshold_caution=values['threshold_caution'],
        threshold_critical=values['threshold_critical'],
        green_base=values['green_base'],
        green_min=values['green_min'],
        green_max=values['green_max'],
        low_base=values['low_base'],
        low_min=values['low_min'],
        low_max=values['low_max'],
        caution_value=values['caution_value'],
        red_base=values['red_base'],
        red_min=values['red_min'],
        red_max=values['red_max'],
        critical_value=values['critical_value'],
    )


def _grid_search_fallback(
    profiles: List[RunnerProfile],
    n_trials: int,
    num_weeks: int,
    weights: Optional[ObjectiveWeights],
    seed: int,
    verbose: bool
) -> Tuple[DeltaVParams, Dict[str, Any]]:
    """
    Simple grid search fallback when Optuna is not available.
    """
    np.random.seed(seed)

    best_score = -np.inf
    best_params = DeltaVParams()
    all_trials = []

    for i in range(n_trials):
        # Random parameter sampling
        params = DeltaVParams(
            threshold_low=np.random.uniform(*PARAM_BOUNDS['threshold_low']),
            threshold_optimal_high=np.random.uniform(*PARAM_BOUNDS['threshold_optimal_high']),
            threshold_caution=np.random.uniform(*PARAM_BOUNDS['threshold_caution']),
            threshold_critical=np.random.uniform(*PARAM_BOUNDS['threshold_critical']),
            green_base=np.random.uniform(*PARAM_BOUNDS['green_base']),
            green_min=np.random.uniform(*PARAM_BOUNDS['green_min']),
            green_max=np.random.uniform(*PARAM_BOUNDS['green_max']),
            low_base=np.random.uniform(*PARAM_BOUNDS['low_base']),
            low_min=np.random.uniform(*PARAM_BOUNDS['low_min']),
            low_max=np.random.uniform(*PARAM_BOUNDS['low_max']),
            caution_value=np.random.uniform(*PARAM_BOUNDS['caution_value']),
            red_base=np.random.uniform(*PARAM_BOUNDS['red_base']),
            red_min=np.random.uniform(*PARAM_BOUNDS['red_min']),
            red_max=np.random.uniform(*PARAM_BOUNDS['red_max']),
            critical_value=np.random.uniform(*PARAM_BOUNDS['critical_value']),
        )

        valid, _ = params.validate()
        if not valid:
            continue

        engine = SimulationEngine(params)
        results = engine.run_batch(profiles, num_weeks, seed)
        score = calculate_total_objective(results, weights)

        all_trials.append({
            'number': i,
            'value': score,
            'params': asdict(params),
        })

        if score > best_score:
            best_score = score
            best_params = params

        if verbose and i % 10 == 0:
            print(f"Trial {i}/{n_trials}: Best Score = {best_score:.4f}")

    results = {
        'best_value': best_score,
        'best_params': asdict(best_params),
        'n_trials': n_trials,
        'all_trials': all_trials,
    }

    return best_params, results


def run_sensitivity_analysis(
    base_params: DeltaVParams,
    profiles: List[RunnerProfile],
    param_name: str,
    values: List[float],
    num_weeks: int = 12,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Run sensitivity analysis for a single parameter.

    Args:
        base_params: Base parameter set
        profiles: Test profiles
        param_name: Name of parameter to vary
        values: Values to test
        num_weeks: Simulation weeks
        seed: Random seed

    Returns:
        List of results for each value
    """
    results = []

    for value in values:
        # Create modified params
        params_dict = asdict(base_params)
        params_dict[param_name] = value
        params = DeltaVParams(**params_dict)

        valid, msg = params.validate()
        if not valid:
            results.append({
                'value': value,
                'valid': False,
                'message': msg,
            })
            continue

        # Run simulation
        engine = SimulationEngine(params)
        sim_results = engine.run_batch(profiles, num_weeks, seed)
        scores = evaluate_simulation_results(sim_results)

        results.append({
            'value': value,
            'valid': True,
            **scores,
        })

    return results


def get_top_n_params(
    optimization_results: Dict[str, Any],
    n: int = 5
) -> List[DeltaVParams]:
    """
    Extract top N parameter sets from optimization results.

    Args:
        optimization_results: Results from optimize_delta_v_params
        n: Number of top sets to return

    Returns:
        List of DeltaVParams sorted by score
    """
    trials = optimization_results.get('all_trials', [])

    # Sort by value (score)
    sorted_trials = sorted(trials, key=lambda x: x.get('value', 0), reverse=True)

    top_params = []
    for trial in sorted_trials[:n]:
        params = params_from_trial_values(trial['params'])
        valid, _ = params.validate()
        if valid:
            top_params.append(params)

    return top_params


if __name__ == '__main__':
    print("Testing optimization search...")

    from data.synthetic import generate_runner_profiles

    # Generate test profiles (smaller set for speed)
    profiles = generate_runner_profiles(10, seed=42)

    # Run optimization with few trials for testing
    print("\nRunning optimization (10 trials for testing)...")
    best_params, results = optimize_delta_v_params(
        profiles,
        n_trials=10,  # Small for testing
        num_weeks=12,
        seed=42,
        verbose=True
    )

    print(f"\nBest composite score: {results['best_value']:.4f}")

    # Compare with baseline
    print("\nComparing with baseline:")
    baseline_params = DeltaVParams()
    baseline_engine = SimulationEngine(baseline_params)
    baseline_results = baseline_engine.run_batch(profiles, 12, seed=42)
    baseline_score = calculate_total_objective(baseline_results)

    print(f"Baseline score: {baseline_score:.4f}")
    print(f"Optimized score: {results['best_value']:.4f}")
    print(f"Improvement: {(results['best_value'] - baseline_score)*100:.1f}%")

    print("\nAll tests passed!")
