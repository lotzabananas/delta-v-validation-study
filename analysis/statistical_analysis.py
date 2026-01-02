"""
Statistical Analysis for Delta V Backtesting Results.

Provides rigorous statistical validation including:
- Power analysis and sample size calculations
- Hypothesis testing (t-tests, paired tests)
- Effect size calculations (Cohen's d)
- Bootstrap confidence intervals
- Overfitting detection
- Multi-seed variance analysis
"""

import sys
import json
import numpy as np
from scipy import stats
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

sys.path.insert(0, '/Users/timmac/Desktop/Delta V backtesting')

from core.delta_v import DeltaVParams
from data.synthetic import generate_runner_profiles
from simulation.engine import SimulationEngine, aggregate_results, SimulationResult
from optimization.objective import evaluate_simulation_results, ObjectiveWeights


@dataclass
class StatisticalTestResult:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    effect_size_interpretation: str
    ci_lower: float
    ci_upper: float
    n1: int
    n2: int
    significant_05: bool
    significant_01: bool


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size.

    Args:
        group1: First group data
        group2: Second group data

    Returns:
        Cohen's d value
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def power_analysis(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
    n: Optional[int] = None
) -> Dict[str, float]:
    """
    Conduct power analysis for two-sample t-test.

    Args:
        effect_size: Expected Cohen's d effect size
        alpha: Significance level
        power: Desired statistical power
        n: Sample size per group (if provided, calculates achieved power)

    Returns:
        Dictionary with power analysis results
    """
    from scipy.stats import norm

    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)

    if n is None:
        # Calculate required sample size
        required_n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        required_n = int(np.ceil(required_n))

        return {
            'required_n_per_group': required_n,
            'required_n_total': required_n * 2,
            'effect_size': effect_size,
            'alpha': alpha,
            'target_power': power,
        }
    else:
        # Calculate achieved power given n
        ncp = effect_size * np.sqrt(n / 2)  # Non-centrality parameter
        achieved_power = 1 - norm.cdf(z_alpha - ncp)

        return {
            'n_per_group': n,
            'effect_size': effect_size,
            'alpha': alpha,
            'achieved_power': achieved_power,
        }


def bootstrap_ci(
    data: np.ndarray,
    statistic: callable = np.mean,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval.

    Args:
        data: Input data array
        statistic: Function to compute statistic
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        seed: Random seed

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper)
    """
    np.random.seed(seed)
    n = len(data)
    bootstrap_stats = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats[i] = statistic(sample)

    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
    point_estimate = statistic(data)

    return point_estimate, ci_lower, ci_upper


def paired_bootstrap_test(
    group1: np.ndarray,
    group2: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Paired bootstrap test for difference in means.

    Args:
        group1: First group (baseline)
        group2: Second group (optimized)
        n_bootstrap: Number of bootstrap samples
        seed: Random seed

    Returns:
        Tuple of (observed_diff, p_value, ci_width)
    """
    np.random.seed(seed)

    observed_diff = np.mean(group2) - np.mean(group1)
    differences = group2 - group1
    n = len(differences)

    # Bootstrap the difference
    bootstrap_diffs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = np.random.choice(differences, size=n, replace=True)
        bootstrap_diffs[i] = np.mean(sample)

    # Two-sided p-value (test if different from 0)
    p_value = np.mean(np.abs(bootstrap_diffs - np.mean(bootstrap_diffs)) >= abs(observed_diff))

    # CI on the difference
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)

    return observed_diff, p_value, (ci_lower, ci_upper)


def run_simulation_multi_seed(
    params: DeltaVParams,
    n_profiles: int,
    n_weeks: int,
    seeds: List[int]
) -> List[List[SimulationResult]]:
    """
    Run simulations with multiple random seeds.

    Args:
        params: Delta V parameters
        n_profiles: Number of profiles per seed
        n_weeks: Simulation weeks
        seeds: List of random seeds

    Returns:
        List of result lists, one per seed
    """
    all_results = []
    engine = SimulationEngine(params)

    for seed in seeds:
        profiles = generate_runner_profiles(n_profiles, seed=seed)
        results = engine.run_batch(profiles, n_weeks, seed=seed)
        all_results.append(results)

    return all_results


def extract_metrics(results: List[SimulationResult]) -> Dict[str, np.ndarray]:
    """Extract key metrics from simulation results."""
    return {
        'growth_ratio': np.array([r.volume_growth_ratio for r in results]),
        'target_reached': np.array([1 if r.target_volume_reached else 0 for r in results]),
        'risk_events': np.array([r.total_risk_events for r in results]),
        'injury_triggered': np.array([1 if r.injury_proxy_triggered else 0 for r in results]),
        'volume_std': np.array([r.volume_change_std for r in results]),
    }


def compare_two_conditions(
    baseline_metrics: Dict[str, np.ndarray],
    optimized_metrics: Dict[str, np.ndarray],
    metric_name: str
) -> StatisticalTestResult:
    """
    Compare two conditions using multiple statistical tests.

    Args:
        baseline_metrics: Baseline condition metrics
        optimized_metrics: Optimized condition metrics
        metric_name: Name of the metric being compared

    Returns:
        StatisticalTestResult with all test outcomes
    """
    baseline = baseline_metrics[metric_name]
    optimized = optimized_metrics[metric_name]

    # Independent samples t-test
    t_stat, p_value = stats.ttest_ind(optimized, baseline)

    # Effect size
    d = cohens_d(optimized, baseline)
    d_interp = interpret_cohens_d(d)

    # Bootstrap CI for difference
    diff = np.mean(optimized) - np.mean(baseline)
    _, ci_lower, ci_upper = bootstrap_ci(
        optimized - np.mean(optimized) + diff / 2,
        statistic=np.mean
    )
    # Adjust CI
    ci_lower = diff - (np.mean(optimized) - ci_lower)
    ci_upper = diff + (ci_upper - np.mean(optimized))

    return StatisticalTestResult(
        test_name=f"T-test: {metric_name}",
        statistic=t_stat,
        p_value=p_value,
        effect_size=d,
        effect_size_interpretation=d_interp,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n1=len(baseline),
        n2=len(optimized),
        significant_05=p_value < 0.05,
        significant_01=p_value < 0.01,
    )


def check_profile_independence(results: List[SimulationResult]) -> Dict[str, Any]:
    """
    Check if synthetic profiles are truly independent.

    Tests for correlations between runner outcomes that might suggest
    non-independence in the synthetic data generation.

    Args:
        results: Simulation results

    Returns:
        Dictionary with independence analysis
    """
    metrics = extract_metrics(results)

    # Calculate correlation matrix between metrics
    metric_names = list(metrics.keys())
    n_metrics = len(metric_names)
    corr_matrix = np.zeros((n_metrics, n_metrics))

    for i, m1 in enumerate(metric_names):
        for j, m2 in enumerate(metric_names):
            if i <= j:
                r, p = stats.pearsonr(metrics[m1], metrics[m2])
                corr_matrix[i, j] = r
                corr_matrix[j, i] = r

    # Check for high correlations (potential non-independence)
    high_correlations = []
    for i in range(n_metrics):
        for j in range(i + 1, n_metrics):
            if abs(corr_matrix[i, j]) > 0.7:
                high_correlations.append({
                    'metric1': metric_names[i],
                    'metric2': metric_names[j],
                    'correlation': corr_matrix[i, j]
                })

    return {
        'correlation_matrix': corr_matrix,
        'metric_names': metric_names,
        'high_correlations': high_correlations,
        'independence_concern': len(high_correlations) > 0,
    }


def check_overfitting(
    train_profiles: List,
    test_profiles: List,
    optimized_params: DeltaVParams,
    baseline_params: DeltaVParams,
    n_weeks: int = 12,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Check for overfitting by comparing train vs test performance.

    Args:
        train_profiles: Profiles used for optimization
        test_profiles: Held-out profiles for testing
        optimized_params: Optimized parameters
        baseline_params: Baseline parameters
        n_weeks: Simulation weeks
        seed: Random seed

    Returns:
        Dictionary with overfitting analysis
    """
    opt_engine = SimulationEngine(optimized_params)
    base_engine = SimulationEngine(baseline_params)

    # Train set performance
    train_opt = opt_engine.run_batch(train_profiles, n_weeks, seed)
    train_base = base_engine.run_batch(train_profiles, n_weeks, seed)

    train_opt_agg = aggregate_results(train_opt)
    train_base_agg = aggregate_results(train_base)

    # Test set performance
    test_opt = opt_engine.run_batch(test_profiles, n_weeks, seed + 1000)
    test_base = base_engine.run_batch(test_profiles, n_weeks, seed + 1000)

    test_opt_agg = aggregate_results(test_opt)
    test_base_agg = aggregate_results(test_base)

    # Calculate improvements
    train_improvement = {
        'growth_ratio': train_opt_agg['mean_growth_ratio'] - train_base_agg['mean_growth_ratio'],
        'target_reached': train_opt_agg['pct_target_reached'] - train_base_agg['pct_target_reached'],
        'risk_reduction': train_base_agg['mean_risk_events'] - train_opt_agg['mean_risk_events'],
    }

    test_improvement = {
        'growth_ratio': test_opt_agg['mean_growth_ratio'] - test_base_agg['mean_growth_ratio'],
        'target_reached': test_opt_agg['pct_target_reached'] - test_base_agg['pct_target_reached'],
        'risk_reduction': test_base_agg['mean_risk_events'] - test_opt_agg['mean_risk_events'],
    }

    # Overfitting ratio (train improvement / test improvement)
    overfitting_ratio = {}
    for key in train_improvement:
        if test_improvement[key] != 0:
            overfitting_ratio[key] = train_improvement[key] / test_improvement[key]
        else:
            overfitting_ratio[key] = float('inf') if train_improvement[key] > 0 else 0

    return {
        'train_improvement': train_improvement,
        'test_improvement': test_improvement,
        'overfitting_ratio': overfitting_ratio,
        'overfitting_detected': any(r > 1.5 for r in overfitting_ratio.values() if r != float('inf')),
        'train_n': len(train_profiles),
        'test_n': len(test_profiles),
    }


def analyze_seed_variance(
    all_seed_results: List[List[SimulationResult]]
) -> Dict[str, Any]:
    """
    Analyze variance across different random seeds.

    Args:
        all_seed_results: Results from multiple seeds

    Returns:
        Dictionary with variance analysis
    """
    # Extract metrics for each seed
    seed_metrics = []
    for results in all_seed_results:
        agg = aggregate_results(results)
        seed_metrics.append({
            'mean_growth_ratio': agg['mean_growth_ratio'],
            'pct_target_reached': agg['pct_target_reached'],
            'mean_risk_events': agg['mean_risk_events'],
            'risk_event_rate': agg['risk_event_rate'],
        })

    # Calculate variance across seeds
    variance_analysis = {}
    for metric in seed_metrics[0].keys():
        values = [sm[metric] for sm in seed_metrics]
        variance_analysis[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'range': np.max(values) - np.min(values),
            'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else 0,
        }

    return {
        'n_seeds': len(all_seed_results),
        'per_seed_metrics': seed_metrics,
        'variance_analysis': variance_analysis,
        'high_variance_metrics': [
            k for k, v in variance_analysis.items() if v['cv'] > 0.2
        ],
    }


def run_full_statistical_analysis(
    n_profiles_list: List[int] = [20, 50, 100],
    n_weeks: int = 12,
    seeds: List[int] = [42, 123, 456, 789, 1000],
    output_dir: str = '/Users/timmac/Desktop/Delta V backtesting/analysis'
) -> Dict[str, Any]:
    """
    Run comprehensive statistical analysis.

    Args:
        n_profiles_list: Sample sizes to test
        n_weeks: Weeks per simulation
        seeds: Random seeds to test
        output_dir: Output directory for results

    Returns:
        Complete analysis results dictionary
    """
    print("=" * 70)
    print("DELTA V BACKTESTING - STATISTICAL ANALYSIS")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load optimized params
    with open('/Users/timmac/Desktop/Delta V backtesting/optimized_params.json', 'r') as f:
        opt_params_dict = json.load(f)

    optimized_params = DeltaVParams(**{k: v for k, v in opt_params_dict.items()
                                        if k in DeltaVParams.__dataclass_fields__})
    baseline_params = DeltaVParams()

    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_profiles_list': n_profiles_list,
            'n_weeks': n_weeks,
            'seeds': seeds,
        },
        'baseline_params': asdict(baseline_params),
        'optimized_params': opt_params_dict,
    }

    # 1. Multi-seed analysis
    print("\n1. MULTI-SEED VARIANCE ANALYSIS")
    print("-" * 40)

    base_seed_results = run_simulation_multi_seed(baseline_params, 20, n_weeks, seeds)
    opt_seed_results = run_simulation_multi_seed(optimized_params, 20, n_weeks, seeds)

    baseline_seed_variance = analyze_seed_variance(base_seed_results)
    optimized_seed_variance = analyze_seed_variance(opt_seed_results)

    print(f"Baseline variance (CV) across {len(seeds)} seeds:")
    for metric, stats_dict in baseline_seed_variance['variance_analysis'].items():
        print(f"  {metric}: mean={stats_dict['mean']:.3f}, CV={stats_dict['cv']:.3f}")

    print(f"\nOptimized variance (CV) across {len(seeds)} seeds:")
    for metric, stats_dict in optimized_seed_variance['variance_analysis'].items():
        print(f"  {metric}: mean={stats_dict['mean']:.3f}, CV={stats_dict['cv']:.3f}")

    results['seed_variance'] = {
        'baseline': baseline_seed_variance,
        'optimized': optimized_seed_variance,
    }

    # 2. Sample size analysis
    print("\n2. SAMPLE SIZE ANALYSIS")
    print("-" * 40)

    sample_size_results = {}
    for n_profiles in n_profiles_list:
        print(f"\nSample size: n={n_profiles}")

        profiles = generate_runner_profiles(n_profiles, seed=42)
        base_engine = SimulationEngine(baseline_params)
        opt_engine = SimulationEngine(optimized_params)

        base_results = base_engine.run_batch(profiles, n_weeks, seed=42)
        opt_results = opt_engine.run_batch(profiles, n_weeks, seed=42)

        base_metrics = extract_metrics(base_results)
        opt_metrics = extract_metrics(opt_results)

        # Statistical comparisons
        tests = {}
        for metric in ['growth_ratio', 'target_reached', 'risk_events']:
            test_result = compare_two_conditions(base_metrics, opt_metrics, metric)
            tests[metric] = asdict(test_result)
            print(f"  {metric}: d={test_result.effect_size:.3f} ({test_result.effect_size_interpretation}), "
                  f"p={test_result.p_value:.4f}, sig={test_result.significant_05}")

        sample_size_results[n_profiles] = {
            'baseline_agg': aggregate_results(base_results),
            'optimized_agg': aggregate_results(opt_results),
            'tests': tests,
        }

    results['sample_size_analysis'] = sample_size_results

    # 3. Power analysis
    print("\n3. POWER ANALYSIS")
    print("-" * 40)

    # Use the n=20 effect sizes for power analysis
    n20_tests = sample_size_results[20]['tests']
    power_results = {}

    for metric in ['growth_ratio', 'target_reached', 'risk_events']:
        observed_d = abs(n20_tests[metric]['effect_size'])
        if observed_d > 0:
            required_n = power_analysis(observed_d, alpha=0.05, power=0.80)
            achieved = power_analysis(observed_d, alpha=0.05, n=20)

            power_results[metric] = {
                'observed_effect_size': observed_d,
                'required_n_for_80_power': required_n['required_n_per_group'],
                'achieved_power_at_n20': achieved['achieved_power'],
            }

            print(f"  {metric}:")
            print(f"    Observed d: {observed_d:.3f}")
            print(f"    Required n for 80% power: {required_n['required_n_per_group']}")
            print(f"    Achieved power at n=20: {achieved['achieved_power']:.1%}")

    results['power_analysis'] = power_results

    # 4. Bootstrap confidence intervals
    print("\n4. BOOTSTRAP CONFIDENCE INTERVALS (n=20)")
    print("-" * 40)

    profiles = generate_runner_profiles(20, seed=42)
    base_engine = SimulationEngine(baseline_params)
    opt_engine = SimulationEngine(optimized_params)

    base_results = base_engine.run_batch(profiles, n_weeks, seed=42)
    opt_results = opt_engine.run_batch(profiles, n_weeks, seed=42)

    bootstrap_results = {}
    for metric in ['growth_ratio', 'risk_events']:
        base_data = extract_metrics(base_results)[metric]
        opt_data = extract_metrics(opt_results)[metric]

        base_est, base_lo, base_hi = bootstrap_ci(base_data)
        opt_est, opt_lo, opt_hi = bootstrap_ci(opt_data)

        bootstrap_results[metric] = {
            'baseline': {'estimate': base_est, 'ci_lower': base_lo, 'ci_upper': base_hi},
            'optimized': {'estimate': opt_est, 'ci_lower': opt_lo, 'ci_upper': opt_hi},
        }

        print(f"  {metric}:")
        print(f"    Baseline: {base_est:.3f} [{base_lo:.3f}, {base_hi:.3f}]")
        print(f"    Optimized: {opt_est:.3f} [{opt_lo:.3f}, {opt_hi:.3f}]")

    results['bootstrap_ci'] = bootstrap_results

    # 5. Profile independence check
    print("\n5. PROFILE INDEPENDENCE CHECK")
    print("-" * 40)

    independence = check_profile_independence(base_results)
    print(f"  High correlations found: {len(independence['high_correlations'])}")
    for corr in independence['high_correlations']:
        print(f"    {corr['metric1']} vs {corr['metric2']}: r={corr['correlation']:.3f}")
    print(f"  Independence concern: {independence['independence_concern']}")

    results['independence_check'] = {
        'high_correlations': independence['high_correlations'],
        'independence_concern': independence['independence_concern'],
    }

    # 6. Overfitting check
    print("\n6. OVERFITTING CHECK")
    print("-" * 40)

    train_profiles = generate_runner_profiles(20, seed=42)  # Same as optimization
    test_profiles = generate_runner_profiles(20, seed=999)  # Different seed

    overfit_analysis = check_overfitting(
        train_profiles, test_profiles,
        optimized_params, baseline_params,
        n_weeks, seed=42
    )

    print(f"  Train improvement:")
    for k, v in overfit_analysis['train_improvement'].items():
        print(f"    {k}: {v:+.3f}")
    print(f"  Test improvement:")
    for k, v in overfit_analysis['test_improvement'].items():
        print(f"    {k}: {v:+.3f}")
    print(f"  Overfitting detected: {overfit_analysis['overfitting_detected']}")

    results['overfitting_check'] = overfit_analysis

    # Save results - simplified version without circular references
    output_path = f"{output_dir}/statistical_results.json"

    # Create a simplified results dict for JSON
    def sanitize_for_json(obj):
        """Recursively sanitize object for JSON serialization."""
        if isinstance(obj, dict):
            new_dict = {}
            for k, v in obj.items():
                # Skip problematic keys
                if isinstance(k, str) and (k.startswith('_') or k in ['correlation_matrix']):
                    continue
                # Convert int keys to strings for JSON
                key = str(k) if isinstance(k, int) else k
                new_dict[key] = sanitize_for_json(v)
            return new_dict
        elif isinstance(obj, list):
            return [sanitize_for_json(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    sanitized_results = sanitize_for_json(results)

    with open(output_path, 'w') as f:
        json.dump(sanitized_results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


def generate_statistical_report(results: Dict[str, Any]) -> str:
    """
    Generate markdown report from statistical analysis results.

    Args:
        results: Results from run_full_statistical_analysis

    Returns:
        Markdown formatted report string
    """
    report = f"""# Statistical Analysis Report: Delta V Backtesting

Generated: {results['timestamp']}

## Executive Summary

This report provides rigorous statistical validation of the Delta V backtesting results,
comparing baseline parameters against optimized parameters discovered through Bayesian optimization.

---

## 1. Sample Size Analysis

### Is n=20 Sufficient?

"""

    # Sample size table
    report += "| Sample Size | Metric | Effect Size (d) | p-value | Significant? | Power at n=20 |\n"
    report += "|-------------|--------|-----------------|---------|--------------|---------------|\n"

    for n in [20, 50, 100]:
        if n in results['sample_size_analysis']:
            for metric in ['growth_ratio', 'target_reached', 'risk_events']:
                test = results['sample_size_analysis'][n]['tests'][metric]
                power = results.get('power_analysis', {}).get(metric, {}).get('achieved_power_at_n20', 'N/A')
                if isinstance(power, float):
                    power_str = f"{power:.1%}"
                else:
                    power_str = str(power)

                sig = "Yes" if test['significant_05'] else "No"
                report += f"| {n} | {metric} | {test['effect_size']:.3f} ({test['effect_size_interpretation']}) | {test['p_value']:.4f} | {sig} | {power_str if n==20 else '-'} |\n"

    report += """
### Power Analysis Interpretation

"""

    if 'power_analysis' in results:
        for metric, pa in results['power_analysis'].items():
            report += f"""**{metric}:**
- Observed effect size (Cohen's d): {pa['observed_effect_size']:.3f}
- Required n per group for 80% power: {pa['required_n_for_80_power']}
- Achieved power at n=20: {pa['achieved_power_at_n20']:.1%}

"""

    # Recommendations
    report += """### Sample Size Recommendations

Based on the power analysis:
"""

    if 'power_analysis' in results:
        max_required = max(pa['required_n_for_80_power'] for pa in results['power_analysis'].values())
        min_power = min(pa['achieved_power_at_n20'] for pa in results['power_analysis'].values())

        if min_power < 0.80:
            report += f"""
- **Current sample size (n=20) is UNDERPOWERED** for detecting all effects
- Minimum achieved power: {min_power:.1%}
- **Recommended sample size: n={max_required}** per condition for 80% power
"""
        else:
            report += f"""
- Current sample size (n=20) provides adequate power (>{min_power:.1%})
- All effects can be reliably detected at this sample size
"""

    report += """
---

## 2. Confidence Intervals

### Bootstrap 95% Confidence Intervals (10,000 iterations)

"""

    if 'bootstrap_ci' in results:
        report += "| Metric | Condition | Estimate | 95% CI Lower | 95% CI Upper |\n"
        report += "|--------|-----------|----------|--------------|---------------|\n"

        for metric, data in results['bootstrap_ci'].items():
            for condition in ['baseline', 'optimized']:
                ci = data[condition]
                report += f"| {metric} | {condition} | {ci['estimate']:.3f} | {ci['ci_lower']:.3f} | {ci['ci_upper']:.3f} |\n"

    report += """
### Interpretation

"""

    if 'bootstrap_ci' in results:
        for metric, data in results['bootstrap_ci'].items():
            base = data['baseline']
            opt = data['optimized']

            # Check for CI overlap
            overlap = not (opt['ci_lower'] > base['ci_upper'] or base['ci_lower'] > opt['ci_upper'])

            if overlap:
                report += f"- **{metric}**: Confidence intervals overlap, suggesting the difference may not be practically significant\n"
            else:
                report += f"- **{metric}**: No CI overlap, strong evidence for a real difference\n"

    report += """
---

## 3. Multi-Seed Variance Analysis

### Variability Across Random Seeds

"""

    if 'seed_variance' in results:
        report += "| Metric | Baseline Mean | Baseline CV | Optimized Mean | Optimized CV |\n"
        report += "|--------|---------------|-------------|----------------|---------------|\n"

        base_var = results['seed_variance']['baseline']['variance_analysis']
        opt_var = results['seed_variance']['optimized']['variance_analysis']

        for metric in base_var.keys():
            report += f"| {metric} | {base_var[metric]['mean']:.3f} | {base_var[metric]['cv']:.3f} | {opt_var[metric]['mean']:.3f} | {opt_var[metric]['cv']:.3f} |\n"

        high_var = results['seed_variance']['baseline'].get('high_variance_metrics', [])
        if high_var:
            report += f"\n**Warning:** High variance (CV > 0.20) detected in: {', '.join(high_var)}\n"
            report += "This suggests results may be sensitive to random seed choice.\n"
        else:
            report += "\nResults show acceptable stability across random seeds (CV < 0.20 for all metrics).\n"

    report += """
---

## 4. Overfitting Analysis

### Train vs Test Performance

"""

    if 'overfitting_check' in results:
        oc = results['overfitting_check']

        report += "| Metric | Train Improvement | Test Improvement | Ratio |\n"
        report += "|--------|-------------------|------------------|-------|\n"

        for metric in oc['train_improvement'].keys():
            train = oc['train_improvement'][metric]
            test = oc['test_improvement'][metric]
            ratio = oc['overfitting_ratio'].get(metric, 'N/A')
            ratio_str = f"{ratio:.2f}" if isinstance(ratio, (int, float)) and ratio != float('inf') else str(ratio)
            report += f"| {metric} | {train:+.3f} | {test:+.3f} | {ratio_str} |\n"

        if oc['overfitting_detected']:
            report += """
**WARNING: Potential overfitting detected!**

The improvement on training data is substantially larger than on held-out test data.
This suggests the optimized parameters may not generalize well to new runners.

**Recommendations:**
1. Use cross-validation during optimization
2. Split data into train/validation/test sets
3. Apply regularization to prevent extreme parameter values
"""
        else:
            report += """
**No significant overfitting detected.**

The improvement generalizes well to held-out test data, suggesting the optimized
parameters capture genuine patterns rather than fitting noise.
"""

    report += """
---

## 5. Profile Independence

"""

    if 'independence_check' in results:
        ic = results['independence_check']

        if ic['independence_concern']:
            report += "**WARNING: Potential dependence between profiles detected!**\n\n"
            report += "High correlations found:\n"
            for corr in ic['high_correlations']:
                report += f"- {corr['metric1']} vs {corr['metric2']}: r={corr['correlation']:.3f}\n"
            report += "\nThis may violate the independence assumption of statistical tests.\n"
        else:
            report += "No concerning correlations detected between runner outcomes.\n"
            report += "The independence assumption for statistical tests appears valid.\n"

    report += """
---

## 6. Summary and Recommendations

### Statistical Validity Assessment

"""

    # Overall assessment
    issues = []

    if 'power_analysis' in results:
        min_power = min(pa['achieved_power_at_n20'] for pa in results['power_analysis'].values())
        if min_power < 0.80:
            issues.append(f"Underpowered study (min power: {min_power:.1%})")

    if 'overfitting_check' in results and results['overfitting_check']['overfitting_detected']:
        issues.append("Potential overfitting detected")

    if 'independence_check' in results and results['independence_check']['independence_concern']:
        issues.append("Profile independence may be violated")

    if 'seed_variance' in results:
        high_var = results['seed_variance']['baseline'].get('high_variance_metrics', [])
        if high_var:
            issues.append(f"High variance across seeds for: {', '.join(high_var)}")

    if issues:
        report += "**Issues identified:**\n"
        for issue in issues:
            report += f"- {issue}\n"
    else:
        report += "**No major statistical issues identified.**\n"

    report += """
### Recommendations for Improving Statistical Rigor

1. **Increase sample size** to at least n=50 profiles for adequate power
2. **Use multiple random seeds** (5-10) and report aggregated results
3. **Implement k-fold cross-validation** in the optimization process
4. **Add regularization** to prevent extreme parameter values
5. **Test on truly independent data** not used in any part of optimization
6. **Report effect sizes** alongside p-values for practical significance

---

*Report generated by Statistical Analysis Agent*
*Delta V Backtesting Framework*
"""

    return report


if __name__ == '__main__':
    print("Running full statistical analysis...")

    # Run analysis
    results = run_full_statistical_analysis(
        n_profiles_list=[20, 50, 100],
        n_weeks=12,
        seeds=[42, 123, 456, 789, 1000]
    )

    # Generate report
    report = generate_statistical_report(results)

    # Save report
    report_path = '/Users/timmac/Desktop/Delta V backtesting/analysis/statistical_report.md'
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\nStatistical report saved to: {report_path}")
    print("\nAnalysis complete!")
