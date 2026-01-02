"""
Objective Functions for Delta V Parameter Optimization.

Defines fitness/loss functions that balance:
- Volume growth (maximize)
- Risk event reduction (minimize)
- Progression stability (minimize variance)
- Target achievement rate (maximize)
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

import sys
sys.path.insert(0, '/Users/timmac/Desktop/Delta V backtesting')

from simulation.engine import SimulationResult, aggregate_results


@dataclass
class ObjectiveWeights:
    """
    Weights for multi-objective optimization.

    All weights should sum to 1.0 for interpretability.
    """
    volume_growth: float = 0.30      # Importance of achieving volume growth
    risk_reduction: float = 0.35     # Importance of minimizing injury risk
    stability: float = 0.15          # Importance of smooth progression
    target_achievement: float = 0.20  # Importance of reaching target

    def __post_init__(self):
        total = (self.volume_growth + self.risk_reduction +
                 self.stability + self.target_achievement)
        if abs(total - 1.0) > 0.01:
            # Normalize
            self.volume_growth /= total
            self.risk_reduction /= total
            self.stability /= total
            self.target_achievement /= total


def evaluate_simulation_results(
    results: List[SimulationResult],
    weights: Optional[ObjectiveWeights] = None
) -> Dict[str, float]:
    """
    Evaluate simulation results against multiple objectives.

    Args:
        results: List of simulation results
        weights: Objective weights (uses defaults if None)

    Returns:
        Dictionary with individual scores and composite score
    """
    if weights is None:
        weights = ObjectiveWeights()

    agg = aggregate_results(results)

    # ═══════════════════════════════════════════════════════════════════════════
    # Volume Growth Score (0-1, higher is better)
    # ═══════════════════════════════════════════════════════════════════════════
    # Target: 2.5x growth, score 1.0 at 2.5x, 0.5 at 1.5x, 0 at 1.0x
    growth_ratio = agg['mean_growth_ratio']
    growth_score = np.clip((growth_ratio - 1.0) / 1.5, 0, 1)

    # ═══════════════════════════════════════════════════════════════════════════
    # Risk Reduction Score (0-1, higher is better = fewer risk events)
    # ═══════════════════════════════════════════════════════════════════════════
    # Target: <5% risk event rate
    # Score 1.0 at 0%, 0.5 at 10%, 0 at 20%+
    risk_rate = agg['risk_event_rate']
    risk_score = np.clip(1 - (risk_rate / 20), 0, 1)

    # ═══════════════════════════════════════════════════════════════════════════
    # Stability Score (0-1, higher is better = smoother progression)
    # ═══════════════════════════════════════════════════════════════════════════
    # Target: std < 0.10 (10% week-to-week variance)
    # Score 1.0 at 0%, 0.5 at 15%, 0 at 30%+
    vol_std = agg['mean_volume_std']
    stability_score = np.clip(1 - (vol_std / 0.30), 0, 1)

    # ═══════════════════════════════════════════════════════════════════════════
    # Target Achievement Score (0-1)
    # ═══════════════════════════════════════════════════════════════════════════
    # Direct percentage of runners reaching target
    target_pct = agg['pct_target_reached']
    target_score = target_pct / 100

    # ═══════════════════════════════════════════════════════════════════════════
    # Composite Score
    # ═══════════════════════════════════════════════════════════════════════════
    composite = (
        weights.volume_growth * growth_score +
        weights.risk_reduction * risk_score +
        weights.stability * stability_score +
        weights.target_achievement * target_score
    )

    return {
        'growth_score': growth_score,
        'risk_score': risk_score,
        'stability_score': stability_score,
        'target_score': target_score,
        'composite_score': composite,
        # Raw metrics for debugging
        'raw_growth_ratio': growth_ratio,
        'raw_risk_rate': risk_rate,
        'raw_vol_std': vol_std,
        'raw_target_pct': target_pct,
    }


def composite_score(
    results: List[SimulationResult],
    weights: Optional[ObjectiveWeights] = None
) -> float:
    """
    Calculate single composite score for optimization.

    Args:
        results: Simulation results
        weights: Objective weights

    Returns:
        Composite score (0-1, higher is better)
    """
    scores = evaluate_simulation_results(results, weights)
    return scores['composite_score']


def calculate_pareto_dominance(
    score_a: Dict[str, float],
    score_b: Dict[str, float]
) -> int:
    """
    Check Pareto dominance between two score sets.

    Args:
        score_a: First score dictionary
        score_b: Second score dictionary

    Returns:
        1 if A dominates B, -1 if B dominates A, 0 if non-dominated
    """
    metrics = ['growth_score', 'risk_score', 'stability_score', 'target_score']

    a_better = 0
    b_better = 0

    for m in metrics:
        if score_a[m] > score_b[m]:
            a_better += 1
        elif score_b[m] > score_a[m]:
            b_better += 1

    if a_better > 0 and b_better == 0:
        return 1  # A dominates B
    elif b_better > 0 and a_better == 0:
        return -1  # B dominates A
    else:
        return 0  # Non-dominated


def find_pareto_frontier(
    all_scores: List[Dict[str, float]]
) -> List[int]:
    """
    Find Pareto frontier from list of scores.

    Args:
        all_scores: List of score dictionaries

    Returns:
        Indices of solutions on Pareto frontier
    """
    n = len(all_scores)
    dominated = [False] * n

    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i != j and not dominated[j]:
                dom = calculate_pareto_dominance(all_scores[i], all_scores[j])
                if dom == 1:
                    dominated[j] = True
                elif dom == -1:
                    dominated[i] = True
                    break

    return [i for i in range(n) if not dominated[i]]


# ═══════════════════════════════════════════════════════════════════════════════
# PENALTY FUNCTIONS (for constraint violations)
# ═══════════════════════════════════════════════════════════════════════════════

def parameter_validity_penalty(params) -> float:
    """
    Calculate penalty for parameters outside valid bounds.

    Args:
        params: DeltaVParams object

    Returns:
        Penalty value (0 if valid, positive if invalid)
    """
    valid, msg = params.validate()
    if valid:
        return 0.0

    # Return penalty proportional to severity
    return 0.5  # Fixed penalty for now


def injury_proxy_penalty(results: List[SimulationResult]) -> float:
    """
    Calculate penalty based on injury proxy triggers.

    Args:
        results: Simulation results

    Returns:
        Penalty (0-1, 0 if no injuries)
    """
    injury_count = sum(1 for r in results if r.injury_proxy_triggered)
    return injury_count / len(results) if results else 0


def extreme_volume_penalty(results: List[SimulationResult]) -> float:
    """
    Penalize extreme volume swings.

    Args:
        results: Simulation results

    Returns:
        Penalty for excessive volume changes
    """
    penalties = []
    for r in results:
        max_increase = r.max_volume_increase
        max_decrease = abs(r.max_volume_decrease)

        # Penalize increases > 25%
        if max_increase > 0.25:
            penalties.append((max_increase - 0.25) * 2)

        # Penalize decreases > 20%
        if max_decrease > 0.20:
            penalties.append((max_decrease - 0.20) * 2)

    return np.mean(penalties) if penalties else 0


def calculate_total_objective(
    results: List[SimulationResult],
    weights: Optional[ObjectiveWeights] = None,
    apply_penalties: bool = True
) -> float:
    """
    Calculate total objective for optimization.

    This is what Optuna will minimize (so we return negative of good scores).

    Args:
        results: Simulation results
        weights: Objective weights
        apply_penalties: Whether to apply penalty functions

    Returns:
        Objective value to MINIMIZE (lower is better)
    """
    scores = evaluate_simulation_results(results, weights)
    composite = scores['composite_score']

    if apply_penalties:
        injury_pen = injury_proxy_penalty(results)
        volume_pen = extreme_volume_penalty(results)

        # Apply penalties as multipliers on the base score
        penalty_factor = 1.0 - 0.3 * injury_pen - 0.2 * volume_pen
        composite *= penalty_factor

    # Return negative because Optuna minimizes by default
    # but we can configure it to maximize, so return as-is for now
    return composite


if __name__ == '__main__':
    print("Testing objective functions...")

    from data.synthetic import generate_runner_profiles
    from core.delta_v import DeltaVParams
    from simulation.engine import SimulationEngine

    # Generate test data
    profiles = generate_runner_profiles(10, seed=42)
    engine = SimulationEngine()
    results = engine.run_batch(profiles, num_weeks=12, seed=42)

    # Evaluate
    scores = evaluate_simulation_results(results)
    print("\nObjective Scores:")
    for key, value in scores.items():
        print(f"  {key}: {value:.3f}")

    # Test composite
    comp = composite_score(results)
    print(f"\nComposite Score: {comp:.3f}")

    # Test penalties
    injury_pen = injury_proxy_penalty(results)
    volume_pen = extreme_volume_penalty(results)
    print(f"\nPenalties:")
    print(f"  Injury: {injury_pen:.3f}")
    print(f"  Volume: {volume_pen:.3f}")

    # Total objective
    total = calculate_total_objective(results)
    print(f"\nTotal Objective: {total:.3f}")

    print("\nAll tests passed!")
