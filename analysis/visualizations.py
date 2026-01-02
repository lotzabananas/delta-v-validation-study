"""
Visualization utilities for Delta V backtesting analysis.

Provides charts for:
- Volume trajectories
- ACWR heatmaps
- Risk event distributions
- Optimization results
- Parameter sensitivity
"""

from typing import List, Optional, Dict, Any, Tuple
import numpy as np

import sys
sys.path.insert(0, '/Users/timmac/Desktop/Delta V backtesting')

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

from simulation.engine import SimulationResult


def plot_volume_trajectories(
    results: List[SimulationResult],
    title: str = "Volume Progression Over Time",
    figsize: Tuple[int, int] = (12, 6),
    show_individual: bool = True,
    show_mean: bool = True,
    show_ci: bool = True,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot volume trajectories for multiple runners.

    Args:
        results: List of simulation results
        title: Plot title
        figsize: Figure size
        show_individual: Show individual runner trajectories
        show_mean: Show mean trajectory
        show_ci: Show 95% confidence interval
        ax: Optional existing axes

    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Get all trajectories
    trajectories = [r.get_volume_trajectory() for r in results]
    max_weeks = max(len(t) for t in trajectories)
    weeks = np.arange(1, max_weeks + 1)

    # Pad shorter trajectories
    padded = []
    for t in trajectories:
        if len(t) < max_weeks:
            t = np.pad(t, (0, max_weeks - len(t)), constant_values=np.nan)
        padded.append(t)
    trajectories = np.array(padded)

    # Plot individual trajectories
    if show_individual:
        for i, result in enumerate(results):
            traj = trajectories[i]
            alpha = 0.3 if len(results) > 5 else 0.5
            ax.plot(weeks, traj, alpha=alpha, linewidth=1,
                   label=result.profile_name if len(results) <= 5 else None)

    # Plot mean and confidence interval
    if show_mean or show_ci:
        mean_traj = np.nanmean(trajectories, axis=0)
        std_traj = np.nanstd(trajectories, axis=0)

        if show_ci:
            ax.fill_between(weeks,
                          mean_traj - 1.96 * std_traj,
                          mean_traj + 1.96 * std_traj,
                          alpha=0.2, color='blue', label='95% CI')

        if show_mean:
            ax.plot(weeks, mean_traj, 'b-', linewidth=2, label='Mean')

    ax.set_xlabel('Week')
    ax.set_ylabel('Weekly Volume (minutes)')
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    return fig


def plot_acwr_trajectories(
    results: List[SimulationResult],
    title: str = "ACWR Over Time",
    figsize: Tuple[int, int] = (12, 6),
    show_zones: bool = True,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot ACWR trajectories with zone overlays.

    Args:
        results: Simulation results
        title: Plot title
        figsize: Figure size
        show_zones: Show ACWR zone backgrounds
        ax: Optional existing axes

    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Get trajectories
    trajectories = [r.get_acwr_trajectory() for r in results]
    max_weeks = max(len(t) for t in trajectories)
    weeks = np.arange(1, max_weeks + 1)

    # Add zone backgrounds
    if show_zones:
        ax.axhspan(0, 0.8, alpha=0.1, color='blue', label='Low')
        ax.axhspan(0.8, 1.3, alpha=0.1, color='green', label='Optimal')
        ax.axhspan(1.3, 1.5, alpha=0.1, color='yellow', label='Caution')
        ax.axhspan(1.5, 2.0, alpha=0.1, color='orange', label='Danger')
        ax.axhspan(2.0, 3.0, alpha=0.1, color='red', label='Critical')

    # Plot trajectories
    for i, result in enumerate(results):
        traj = trajectories[i]
        ax.plot(range(1, len(traj) + 1), traj, alpha=0.5, linewidth=1)

    # Plot mean
    padded = []
    for t in trajectories:
        if len(t) < max_weeks:
            t = np.pad(t, (0, max_weeks - len(t)), constant_values=np.nan)
        padded.append(t)
    mean_acwr = np.nanmean(padded, axis=0)
    ax.plot(weeks, mean_acwr, 'k-', linewidth=2, label='Mean ACWR')

    ax.set_xlabel('Week')
    ax.set_ylabel('ACWR')
    ax.set_title(title)
    ax.set_ylim(0, 2.5)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    return fig


def plot_acwr_heatmap(
    results: List[SimulationResult],
    title: str = "ACWR Heatmap by Runner and Week",
    figsize: Tuple[int, int] = (14, 8),
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Create heatmap of ACWR values.

    Args:
        results: Simulation results
        title: Plot title
        figsize: Figure size
        ax: Optional existing axes

    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Build matrix
    trajectories = [r.get_acwr_trajectory() for r in results]
    max_weeks = max(len(t) for t in trajectories)

    matrix = np.full((len(results), max_weeks), np.nan)
    for i, t in enumerate(trajectories):
        matrix[i, :len(t)] = t

    # Custom colormap: blue -> green -> yellow -> orange -> red
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'acwr',
        [(0, 'blue'), (0.4, 'green'), (0.65, 'yellow'),
         (0.75, 'orange'), (1, 'red')]
    )

    im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=0, vmax=2.0)

    ax.set_xlabel('Week')
    ax.set_ylabel('Runner')
    ax.set_title(title)
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels([r.profile_name[:15] for r in results])
    ax.set_xticks(range(max_weeks))
    ax.set_xticklabels(range(1, max_weeks + 1))

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('ACWR')

    return fig


def plot_risk_distribution(
    results: List[SimulationResult],
    title: str = "Risk Event Distribution",
    figsize: Tuple[int, int] = (10, 6),
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot distribution of risk events across runners.

    Args:
        results: Simulation results
        title: Plot title
        figsize: Figure size
        ax: Optional existing axes

    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Extract risk data
    names = [r.profile_name[:15] for r in results]
    risk_events = [r.total_risk_events for r in results]
    max_consecutive = [r.max_consecutive_risk for r in results]
    injury_proxy = [r.injury_proxy_triggered for r in results]

    x = np.arange(len(results))
    width = 0.35

    bars1 = ax.bar(x - width/2, risk_events, width, label='Total Risk Weeks')
    bars2 = ax.bar(x + width/2, max_consecutive, width, label='Max Consecutive')

    # Highlight injury proxy triggers
    for i, triggered in enumerate(injury_proxy):
        if triggered:
            ax.bar(x[i] - width/2, risk_events[i], width,
                  color='red', alpha=0.7)
            ax.bar(x[i] + width/2, max_consecutive[i], width,
                  color='darkred', alpha=0.7)

    ax.set_xlabel('Runner')
    ax.set_ylabel('Weeks')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()

    # Add injury proxy legend
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='Injury Proxy Triggered')
    ]
    ax.legend(handles=ax.get_legend_handles_labels()[0] + legend_elements,
             loc='upper right')

    plt.tight_layout()
    return fig


def plot_optimization_results(
    optimization_results: Dict[str, Any],
    title: str = "Optimization Progress",
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Plot optimization progress and best parameters.

    Args:
        optimization_results: Results from optimize_delta_v_params
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    trials = optimization_results.get('all_trials', [])
    values = [t['value'] for t in trials]
    trial_nums = [t['number'] for t in trials]

    # Score progression
    ax1 = axes[0]
    ax1.plot(trial_nums, values, 'b.', alpha=0.5, markersize=4)

    # Running best
    running_best = np.maximum.accumulate(values)
    ax1.plot(trial_nums, running_best, 'r-', linewidth=2, label='Best So Far')

    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Composite Score')
    ax1.set_title('Optimization Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Best parameters
    ax2 = axes[1]
    best_params = optimization_results.get('best_params', {})

    # Select key parameters to display
    key_params = ['green_max', 'low_max', 'threshold_low',
                  'threshold_optimal_high', 'red_base', 'critical_value']
    param_values = [best_params.get(p, 0) for p in key_params]

    bars = ax2.barh(key_params, param_values, color='steelblue')
    ax2.set_xlabel('Value')
    ax2.set_title('Best Parameters')

    # Add value labels
    for bar, val in zip(bars, param_values):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center')

    plt.suptitle(title)
    plt.tight_layout()
    return fig


def plot_comparison(
    baseline_results: List[SimulationResult],
    optimized_results: List[SimulationResult],
    title: str = "Baseline vs Optimized Comparison",
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """
    Compare baseline and optimized parameter performance.

    Args:
        baseline_results: Results with baseline parameters
        optimized_results: Results with optimized parameters
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Volume trajectories comparison
    ax1 = axes[0, 0]
    for r in baseline_results:
        t = r.get_volume_trajectory()
        ax1.plot(range(1, len(t)+1), t, 'b-', alpha=0.3)
    for r in optimized_results:
        t = r.get_volume_trajectory()
        ax1.plot(range(1, len(t)+1), t, 'r-', alpha=0.3)

    ax1.set_xlabel('Week')
    ax1.set_ylabel('Volume (min)')
    ax1.set_title('Volume Trajectories')
    ax1.legend(['Baseline', 'Optimized'], loc='upper left')

    # ACWR comparison
    ax2 = axes[0, 1]
    baseline_acwrs = [r.get_acwr_trajectory().mean() for r in baseline_results]
    optimized_acwrs = [r.get_acwr_trajectory().mean() for r in optimized_results]

    x = np.arange(len(baseline_results))
    width = 0.35
    ax2.bar(x - width/2, baseline_acwrs, width, label='Baseline', alpha=0.7)
    ax2.bar(x + width/2, optimized_acwrs, width, label='Optimized', alpha=0.7)
    ax2.axhline(y=0.8, color='g', linestyle='--', alpha=0.5)
    ax2.axhline(y=1.3, color='y', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Runner')
    ax2.set_ylabel('Mean ACWR')
    ax2.set_title('Mean ACWR Comparison')
    ax2.legend()

    # Risk events comparison
    ax3 = axes[1, 0]
    baseline_risk = [r.total_risk_events for r in baseline_results]
    optimized_risk = [r.total_risk_events for r in optimized_results]

    ax3.bar(x - width/2, baseline_risk, width, label='Baseline', alpha=0.7)
    ax3.bar(x + width/2, optimized_risk, width, label='Optimized', alpha=0.7)
    ax3.set_xlabel('Runner')
    ax3.set_ylabel('Risk Events')
    ax3.set_title('Risk Events Comparison')
    ax3.legend()

    # Summary metrics
    ax4 = axes[1, 1]

    metrics = ['Avg Growth', 'Avg Risk Events', 'Target Reached %']
    baseline_vals = [
        np.mean([r.volume_growth_ratio for r in baseline_results]),
        np.mean([r.total_risk_events for r in baseline_results]),
        np.mean([r.target_volume_reached for r in baseline_results]) * 100,
    ]
    optimized_vals = [
        np.mean([r.volume_growth_ratio for r in optimized_results]),
        np.mean([r.total_risk_events for r in optimized_results]),
        np.mean([r.target_volume_reached for r in optimized_results]) * 100,
    ]

    x = np.arange(len(metrics))
    ax4.bar(x - width/2, baseline_vals, width, label='Baseline', alpha=0.7)
    ax4.bar(x + width/2, optimized_vals, width, label='Optimized', alpha=0.7)
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.set_title('Summary Metrics')
    ax4.legend()

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def plot_sensitivity_analysis(
    sensitivity_results: List[Dict[str, Any]],
    param_name: str,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot sensitivity analysis results.

    Args:
        sensitivity_results: Results from run_sensitivity_analysis
        param_name: Name of parameter varied
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    valid_results = [r for r in sensitivity_results if r.get('valid', False)]

    if not valid_results:
        ax.text(0.5, 0.5, 'No valid results',
               ha='center', va='center', transform=ax.transAxes)
        return fig

    values = [r['value'] for r in valid_results]
    scores = [r['composite_score'] for r in valid_results]
    growth = [r['growth_score'] for r in valid_results]
    risk = [r['risk_score'] for r in valid_results]

    ax.plot(values, scores, 'b-o', linewidth=2, markersize=8, label='Composite')
    ax.plot(values, growth, 'g--', linewidth=1.5, label='Growth')
    ax.plot(values, risk, 'r--', linewidth=1.5, label='Risk')

    ax.set_xlabel(param_name)
    ax.set_ylabel('Score')
    ax.set_title(title or f'Sensitivity Analysis: {param_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def create_summary_dashboard(
    results: List[SimulationResult],
    title: str = "Simulation Summary Dashboard",
    figsize: Tuple[int, int] = (16, 12),
) -> plt.Figure:
    """
    Create comprehensive summary dashboard.

    Args:
        results: Simulation results
        title: Dashboard title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)

    # Layout: 2x3 grid
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)

    # Volume trajectories
    plot_volume_trajectories(results, ax=ax1, title="Volume Progression")

    # ACWR trajectories
    plot_acwr_trajectories(results, ax=ax2, title="ACWR Over Time")

    # Risk distribution
    plot_risk_distribution(results, ax=ax3, title="Risk Events")

    # Final volume distribution
    final_vols = [r.final_volume for r in results]
    ax4.hist(final_vols, bins=10, edgecolor='black', alpha=0.7)
    ax4.axvline(np.mean(final_vols), color='r', linestyle='--',
               label=f'Mean: {np.mean(final_vols):.0f}')
    ax4.set_xlabel('Final Volume (min)')
    ax4.set_ylabel('Count')
    ax4.set_title('Final Volume Distribution')
    ax4.legend()

    # Growth ratio distribution
    growth_ratios = [r.volume_growth_ratio for r in results]
    ax5.hist(growth_ratios, bins=10, edgecolor='black', alpha=0.7)
    ax5.axvline(np.mean(growth_ratios), color='r', linestyle='--',
               label=f'Mean: {np.mean(growth_ratios):.2f}x')
    ax5.set_xlabel('Growth Ratio')
    ax5.set_ylabel('Count')
    ax5.set_title('Volume Growth Distribution')
    ax5.legend()

    # Summary text
    ax6.axis('off')
    summary_text = f"""
    SIMULATION SUMMARY
    ==================

    Runners: {len(results)}
    Weeks: {len(results[0].weeks) if results else 0}

    Volume:
      Initial (mean): {np.mean([r.initial_volume for r in results]):.0f} min
      Final (mean): {np.mean(final_vols):.0f} min
      Growth ratio: {np.mean(growth_ratios):.2f}x

    Risk:
      Total risk events: {sum(r.total_risk_events for r in results)}
      Runners with injury proxy: {sum(r.injury_proxy_triggered for r in results)}

    Success:
      Target reached: {sum(r.target_volume_reached for r in results)}/{len(results)}
      ({100*sum(r.target_volume_reached for r in results)/len(results):.0f}%)
    """
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    print("Testing visualizations...")

    from data.synthetic import generate_runner_profiles
    from core.delta_v import DeltaVParams
    from simulation.engine import SimulationEngine

    # Generate test data
    profiles = generate_runner_profiles(10, seed=42)
    engine = SimulationEngine()
    results = engine.run_batch(profiles, num_weeks=12, seed=42)

    # Test each visualization
    print("Creating visualizations...")

    fig1 = plot_volume_trajectories(results)
    fig1.savefig('/Users/timmac/Desktop/Delta V backtesting/test_volume.png', dpi=100)
    print("  Saved test_volume.png")

    fig2 = plot_acwr_trajectories(results)
    fig2.savefig('/Users/timmac/Desktop/Delta V backtesting/test_acwr.png', dpi=100)
    print("  Saved test_acwr.png")

    fig3 = plot_acwr_heatmap(results)
    fig3.savefig('/Users/timmac/Desktop/Delta V backtesting/test_heatmap.png', dpi=100)
    print("  Saved test_heatmap.png")

    fig4 = create_summary_dashboard(results)
    fig4.savefig('/Users/timmac/Desktop/Delta V backtesting/test_dashboard.png', dpi=100)
    print("  Saved test_dashboard.png")

    plt.close('all')
    print("\nAll visualization tests passed!")
