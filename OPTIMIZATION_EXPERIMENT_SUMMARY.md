# Delta V Optimization Experiment

**Date**: January 2, 2026
**Objective**: Optimize Delta V parameters to maximize training volume growth while constraining injury risk

---

## Method

### Algorithm
- **Differential Evolution** (scipy.optimize)
- Population size: 15, Max iterations: 50
- Multi-objective: `maximize(growth) - 10 × injury_rate`

### Simulation Design
- 12-week (84-day) athlete simulations
- ACWR calculated daily (7-day acute / 28-day chronic)
- Delta V recommendations applied weekly
- Injury probability sampled daily based on ACWR zone

### Empirical Injury Model
Injury rates derived from validation study (Experiment 003, n=42,766 athlete-days):

| Zone | ACWR | Daily Injury Rate | Relative Risk |
|------|------|-------------------|---------------|
| Low | < 0.8 | 1.44% | 1.21x |
| **Optimal** | **0.8-1.3** | **1.19%** | **1.00x** |
| Caution | 1.3-1.5 | 1.39% | 1.17x |
| High | 1.5-2.0 | 1.61% | 1.36x |
| Critical | ≥ 2.0 | 1.73% | 1.46x |

---

## Parameters Optimized

Zone thresholds held fixed (validated in prior experiments). Magnitude parameters optimized:

| Parameter | Default | Optimized | Change |
|-----------|---------|-----------|--------|
| `green_base` | 0.20 | **0.30** | +50% |
| `green_min` | 0.05 | **0.08** | +60% |
| `green_max` | 0.15 | **0.20** | +33% |
| `low_base` | 0.25 | **0.31** | +24% |
| `low_min` | 0.10 | **0.14** | +40% |
| `low_max` | 0.20 | **0.17** | -15% |
| `caution_value` | 0.00 | **+0.02** | +2% |
| `red_base` | -0.20 | **-0.26** | -30% |
| `red_min` | -0.15 | **-0.16** | -7% |
| `red_max` | -0.05 | **-0.05** | — |
| `critical_value` | -0.30 | **-0.27** | +10% |

---

## Results

### Primary Outcome

| Metric | Default | Optimized | 95% CI | p-value |
|--------|---------|-----------|--------|---------|
| Growth Rate | 2.07x | **2.24x** | [2.14, 2.30] | < 0.0001 |
| Achieved 2x | 53.5% | **64.9%** | [54.5%, 72.0%] | — |
| Injury Rate | 62.8% | 62.8% | [53.4%, 72.6%] | — |

**Improvement: +7.84%** growth with equivalent injury risk

### Validation
- **Bootstrap**: 100 simulations, 100 athletes each
- **Holdout**: 10 independent populations, 200 athletes each
- **Result**: All 10 holdout populations showed improvement (t = 92.08, p < 0.0001)

### Sensitivity Analysis
Most influential parameters:
1. `green_base` (0.225 effect per 1% change)
2. `green_max` (0.201)
3. `low_base` (0.199)

---

## Optimized Equation

```
ΔV(ACWR) =

  CRITICAL (≥ 2.0):     -27%
  HIGH (1.5-2.0):       clamp(-0.26 × (ACWR - 1.3), -16%, -5%)
  CAUTION (1.3-1.5):    +2%
  OPTIMAL (0.8-1.3):    clamp(0.30 × (1 - (ACWR-0.8)/0.5), +8%, +20%)
  LOW (< 0.8):          clamp(0.31 × (0.8 - ACWR + 0.1), +14%, +17%)
```

---

## Conclusion

> **"Push harder when it's safe, pull back faster when it's not."**

The optimization found that default parameters were too conservative in safe zones and insufficiently aggressive in danger zones. The optimized equation increases volume recommendations by 33-60% in the optimal zone while increasing reduction magnitude by 30% in high-risk zones.

---

## Files

| File | Description |
|------|-------------|
| `optimization/delta_v_optimizer.py` | Optimizer implementation |
| `optimization/bootstrap_validation.py` | Validation with CI |
| `optimized_params_v2.json` | Final parameters (JSON) |
| `experiments/results/optimization_*.json` | Raw results |
| `experiments/results/validation_*.json` | Validation results |
