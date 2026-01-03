# Optimized Delta V Equation

**Date**: January 2, 2026
**Optimization Method**: Differential Evolution with empirical injury risk model

---

## The Optimized Piecewise Function

```
ΔV(ACWR) =

┌─ CRITICAL (ACWR ≥ 2.0):
│    ΔV = -27%
│
├─ RED/HIGH RISK (1.5 ≤ ACWR < 2.0):
│    ΔV = clamp(-0.26 × (ACWR - 1.3), -16%, -5%)
│
├─ CAUTION (1.3 ≤ ACWR < 1.5):
│    ΔV = +2%  (slight increase allowed)
│
├─ OPTIMAL/GREEN (0.8 ≤ ACWR < 1.3):
│    ΔV = clamp(0.30 × (1 - (ACWR - 0.8) / 0.5), +8%, +20%)
│
└─ LOW (ACWR < 0.8):
     ΔV = clamp(0.31 × (0.8 - ACWR + 0.1), +14%, +17%)
```

---

## Parameter Comparison: Default vs Optimized

| Parameter | Default | Optimized | Change |
|-----------|---------|-----------|--------|
| **Zone Boundaries** | | | |
| threshold_low | 0.80 | 0.80 | — |
| threshold_optimal_high | 1.30 | 1.30 | — |
| threshold_caution | 1.50 | 1.50 | — |
| threshold_critical | 2.00 | 2.00 | — |
| **Green Zone (Optimal)** | | | |
| green_base | 0.20 | **0.30** | +50% |
| green_min | 0.05 | **0.08** | +60% |
| green_max | 0.15 | **0.20** | +33% |
| **Low Zone** | | | |
| low_base | 0.25 | **0.31** | +24% |
| low_min | 0.10 | **0.14** | +44% |
| low_max | 0.20 | **0.17** | -17% |
| **Caution Zone** | | | |
| caution_value | 0.00 | **+0.02** | +2% |
| **Red Zone (High Risk)** | | | |
| red_base | -0.20 | **-0.26** | -30% |
| red_min | -0.15 | **-0.16** | -9% |
| red_max | -0.05 | **-0.05** | — |
| **Critical Zone** | | | |
| critical_value | -0.30 | **-0.27** | +10% |

---

## Key Insights from Optimization

### 1. Be MORE AGGRESSIVE in Safe Zones
- Green zone increases up to **+20%** (was +15%)
- Low zone minimum increase is **+14%** (was +10%)
- Even caution zone allows **+2%** increase (was 0%)

### 2. Be MORE CONSERVATIVE in Danger Zones
- Red zone reductions up to **-16%** (was -15%)
- Red zone base multiplier increased by **30%** (more reduction at same ACWR)

### 3. Slightly Softer Critical Response
- Critical value is **-27%** (was -30%)
- This prevents over-correction that might swing ACWR too far

---

## Simulation Results

| Metric | Default | Optimized | 95% CI | Improvement |
|--------|---------|-----------|--------|-------------|
| Growth Rate | 2.01x | **2.21x** | [2.14, 2.30] | +7.8% |
| Achieved 2x Target | 45.6% | **63.4%** | [54.5%, 72.0%] | +17.8 pts |
| Achieved 1.5x Target | 97.4% | **97.8%** | — | +0.4 pts |
| Injury Rate | 66.6% | 63.2% | [53.4%, 72.6%] | — |

**Note**: Injury rate stayed approximately constant because it's determined by zone boundaries and empirical injury rates. The optimization maximized growth within the injury risk constraint.

---

## Statistical Validation

### Bootstrap Analysis (n=100 simulations)
- Growth rate: **2.214x** (95% CI: [2.140, 2.303])
- Achieved 2x target: **63.4%** (95% CI: [54.5%, 72.0%])

### Holdout Validation (10 independent populations, 200 athletes each)
| Metric | Default | Optimized |
|--------|---------|-----------|
| Growth Rate | 2.074x | **2.237x** |
| Achieved 2x | 53.5% | **64.9%** |

**Statistical Test**:
- Mean improvement: **+7.84%**
- t-statistic: 92.077
- **p-value: < 0.0001**
- All 10 holdout populations showed improvement

### Parameter Sensitivity Ranking
Most influential parameters (effect per 1% change):
1. `green_base` (0.225) - Optimal zone multiplier
2. `green_max` (0.201) - Maximum optimal increase
3. `low_base` (0.199) - Low zone multiplier

---

## Practical Recommendations

### ACWR-Based Volume Adjustments

| ACWR | Zone | Recommendation |
|------|------|----------------|
| < 0.6 | Very Low | Increase +17% |
| 0.6 - 0.8 | Low | Increase +14% to +17% |
| 0.8 - 1.0 | Optimal (lower) | Increase +15% to +20% |
| 1.0 - 1.3 | Optimal (upper) | Increase +8% to +15% |
| 1.3 - 1.5 | Caution | Increase +2% |
| 1.5 - 1.7 | High Risk | Decrease -5% to -10% |
| 1.7 - 2.0 | High Risk | Decrease -10% to -16% |
| ≥ 2.0 | Critical | Decrease -27% |

---

## Mathematical Formulation

### Optimal Zone (0.8 ≤ ACWR < 1.3)

```
ΔV = max(0.08, min(0.20, 0.30 × (1 - (ACWR - 0.8) / 0.5)))
```

At ACWR = 0.8: ΔV = 0.30 × 1.0 = 30% → clamped to **+20%**
At ACWR = 1.0: ΔV = 0.30 × 0.6 = 18% → **+18%**
At ACWR = 1.3: ΔV = 0.30 × 0.0 = 0% → clamped to **+8%**

### Red Zone (1.5 ≤ ACWR < 2.0)

```
ΔV = max(-0.16, min(-0.05, -0.26 × (ACWR - 1.3)))
```

At ACWR = 1.5: ΔV = -0.26 × 0.2 = -5.2% → **-5%**
At ACWR = 1.7: ΔV = -0.26 × 0.4 = -10.4% → **-10%**
At ACWR = 2.0: ΔV = -0.26 × 0.7 = -18.2% → clamped to **-16%**

---

## JSON Configuration

```json
{
  "threshold_low": 0.8,
  "threshold_optimal_high": 1.3,
  "threshold_caution": 1.5,
  "threshold_critical": 2.0,
  "green_base": 0.30,
  "green_min": 0.08,
  "green_max": 0.20,
  "low_base": 0.31,
  "low_min": 0.14,
  "low_max": 0.17,
  "caution_value": 0.02,
  "red_base": -0.26,
  "red_min": -0.16,
  "red_max": -0.05,
  "critical_value": -0.27
}
```

---

## Validation Notes

1. **Zone boundaries kept fixed**: The 0.8, 1.3, 1.5, 2.0 thresholds were validated in Experiment 003 and kept constant during optimization.

2. **Empirical injury rates used**: The optimization used real injury probabilities by zone from our validation study (RR = 1.30 at ACWR ≥ 1.5).

3. **12-week simulation**: Athletes were simulated for 84 days to evaluate long-term growth and injury outcomes.

4. **Multi-objective optimization**: Maximized growth while penalizing injuries (weight = 10).

---

## Conclusion

The optimized Delta V equation recommends:

> **"Push harder when it's safe, pull back faster when it's not."**

- In the optimal zone (ACWR 0.8-1.3): Increase up to **+20%** per week
- In the danger zone (ACWR 1.5+): Decrease by **-16%** or more
- This achieves **+7.3% better growth** with the same injury risk
