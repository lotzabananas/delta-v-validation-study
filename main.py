#!/usr/bin/env python3
"""
Delta V Backtesting Framework - CLI Entry Point

Usage:
    python main.py backtest [--profiles N] [--weeks W] [--seed S]
    python main.py optimize [--trials N] [--profiles P] [--weeks W]
    python main.py compare [--trials N]
    python main.py test
"""

import sys
import argparse
import json
from dataclasses import asdict

sys.path.insert(0, '/Users/timmac/Desktop/Delta V backtesting')

from core.delta_v import DeltaVParams
from data.synthetic import generate_runner_profiles
from simulation.engine import SimulationEngine, aggregate_results
from optimization.objective import evaluate_simulation_results, ObjectiveWeights
from optimization.search import optimize_delta_v_params
from analysis.reports import generate_backtest_report, generate_comparison_report


def run_backtest(n_profiles: int = 20, n_weeks: int = 12, seed: int = 42):
    """Run baseline backtest."""
    print(f"Running backtest with {n_profiles} profiles over {n_weeks} weeks...")

    profiles = generate_runner_profiles(n_profiles, seed=seed)
    params = DeltaVParams()
    engine = SimulationEngine(params, verbose=True)

    results = engine.run_batch(profiles, n_weeks, seed=seed)

    report = generate_backtest_report(results, params)
    print(report)

    # Evaluate scores
    scores = evaluate_simulation_results(results)
    print("\nObjective Scores:")
    for key, value in scores.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")

    return results


def run_optimization(
    n_trials: int = 100,
    n_profiles: int = 20,
    n_weeks: int = 12,
    seed: int = 42
):
    """Run parameter optimization."""
    print(f"Running optimization: {n_trials} trials, {n_profiles} profiles, {n_weeks} weeks...")

    profiles = generate_runner_profiles(n_profiles, seed=seed)

    weights = ObjectiveWeights(
        volume_growth=0.30,
        risk_reduction=0.35,
        stability=0.15,
        target_achievement=0.20
    )

    best_params, results = optimize_delta_v_params(
        profiles,
        n_trials=n_trials,
        num_weeks=n_weeks,
        weights=weights,
        seed=seed,
        verbose=True
    )

    # Save results
    output_path = '/Users/timmac/Desktop/Delta V backtesting/optimized_params.json'
    with open(output_path, 'w') as f:
        json.dump(results['best_params'], f, indent=2)

    print(f"\nOptimized parameters saved to: {output_path}")
    print(f"Best score: {results['best_value']:.4f}")

    return best_params, results


def run_comparison(n_trials: int = 100, n_profiles: int = 20, n_weeks: int = 12, seed: int = 42):
    """Run optimization and compare with baseline."""
    print("Running comparison analysis...")

    profiles = generate_runner_profiles(n_profiles, seed=seed)

    # Baseline
    baseline_params = DeltaVParams()
    baseline_engine = SimulationEngine(baseline_params)
    baseline_results = baseline_engine.run_batch(profiles, n_weeks, seed=seed)

    # Optimize
    weights = ObjectiveWeights()
    best_params, opt_results = optimize_delta_v_params(
        profiles,
        n_trials=n_trials,
        num_weeks=n_weeks,
        weights=weights,
        seed=seed,
        verbose=True
    )

    # Run optimized
    optimized_engine = SimulationEngine(best_params)
    optimized_results = optimized_engine.run_batch(profiles, n_weeks, seed=seed)

    # Generate comparison report
    report = generate_comparison_report(
        baseline_results, optimized_results,
        baseline_params, best_params
    )
    print(report)

    return baseline_results, optimized_results, best_params


def run_tests():
    """Run all module tests."""
    print("Running tests...\n")

    # Test core metrics
    print("Testing core metrics...")
    from core.metrics import calculate_trimp, calculate_acwr
    import numpy as np

    trimp = calculate_trimp(60, 140, 60, 180, gender='male')
    assert trimp > 0, "TRIMP should be positive"
    print(f"  TRIMP test passed: {trimp:.2f}")

    daily_trimp = np.random.normal(50, 10, 35)
    acwr, _, _ = calculate_acwr(daily_trimp)
    assert 0 <= acwr <= 5, "ACWR should be in reasonable range"
    print(f"  ACWR test passed: {acwr:.2f}")

    # Test Delta V
    print("\nTesting Delta V equation...")
    from core.delta_v import calculate_delta_v, DeltaVParams

    params = DeltaVParams()
    delta_v, zone, _ = calculate_delta_v(1.0, params)
    assert zone == 'optimal', f"ACWR 1.0 should be optimal, got {zone}"
    assert delta_v > 0, "Delta V should be positive in optimal zone"
    print(f"  Delta V test passed: {delta_v*100:+.1f}% ({zone})")

    # Test simulation
    print("\nTesting simulation engine...")
    from data.synthetic import generate_runner_profiles
    from simulation.engine import SimulationEngine

    profiles = generate_runner_profiles(3, seed=42)
    engine = SimulationEngine()
    results = engine.run_batch(profiles, num_weeks=4, seed=42)
    assert len(results) == 3, "Should have 3 results"
    print(f"  Simulation test passed: {len(results)} runners simulated")

    # Test optimization objective
    print("\nTesting optimization objectives...")
    from optimization.objective import evaluate_simulation_results

    scores = evaluate_simulation_results(results)
    assert 'composite_score' in scores, "Should have composite score"
    print(f"  Objective test passed: composite = {scores['composite_score']:.4f}")

    print("\n" + "="*50)
    print("ALL TESTS PASSED!")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description='Delta V Backtesting Framework')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Backtest command
    bt_parser = subparsers.add_parser('backtest', help='Run baseline backtest')
    bt_parser.add_argument('--profiles', type=int, default=20, help='Number of profiles')
    bt_parser.add_argument('--weeks', type=int, default=12, help='Simulation weeks')
    bt_parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Optimize command
    opt_parser = subparsers.add_parser('optimize', help='Run parameter optimization')
    opt_parser.add_argument('--trials', type=int, default=100, help='Optimization trials')
    opt_parser.add_argument('--profiles', type=int, default=20, help='Number of profiles')
    opt_parser.add_argument('--weeks', type=int, default=12, help='Simulation weeks')
    opt_parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Compare command
    cmp_parser = subparsers.add_parser('compare', help='Run comparison analysis')
    cmp_parser.add_argument('--trials', type=int, default=100, help='Optimization trials')
    cmp_parser.add_argument('--profiles', type=int, default=20, help='Number of profiles')
    cmp_parser.add_argument('--weeks', type=int, default=12, help='Simulation weeks')
    cmp_parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Test command
    subparsers.add_parser('test', help='Run tests')

    args = parser.parse_args()

    if args.command == 'backtest':
        run_backtest(args.profiles, args.weeks, args.seed)
    elif args.command == 'optimize':
        run_optimization(args.trials, args.profiles, args.weeks, args.seed)
    elif args.command == 'compare':
        run_comparison(args.trials, args.profiles, args.weeks, args.seed)
    elif args.command == 'test':
        run_tests()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
