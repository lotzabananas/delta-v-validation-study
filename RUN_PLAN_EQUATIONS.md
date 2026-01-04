# Complete Run Plan Equations: Implementation Specification

**Date**: January 4, 2026
**Purpose**: Comprehensive equation set for MAF/Zone 2 → 80/20 running app
**Status**: Implementation specification with critiques and refinements

---

## Executive Summary

This document specifies the complete set of equations needed to dynamically adjust a running plan. The system integrates:
- **Base metrics** (MAF HR, initial volume)
- **Load quantification** (TRIMP, EWMA, ACWR)
- **Volume adjustment** (Delta V with validated parameters)
- **Frequency modulation** (days per week)
- **Workout distribution** (type splits)
- **Scheduling optimization** (spacing and order)

### Methodological Notes and Critiques

| Equation | Original Proposal | Critique | Refinement |
|----------|-------------------|----------|------------|
| MAF HR | 180 - Age ± adjustments | Heuristic, not physiologically derived; doesn't account for trained vs untrained cardiac drift | Add fitness-level scaling; integrate with observed HR zones from HealthKit |
| Initial Volume | Binary 60/120 min | Too coarse; doesn't leverage training history | Continuous function based on recent activity + fitness assessment |
| Delta V | Default parameters | Our optimization study found +7.8% better outcomes with refined parameters | Use validated optimized parameters (see OPTIMIZED_EQUATION.md) |
| BMI Modifier | Simple >30 threshold | BMI is crude; doesn't distinguish muscle vs fat | Use as secondary signal; prefer HR-based stress detection |
| Frequency | Floor division | Ignores recovery capacity and individual variation | Add recovery-quality modifier |
| Day Assignment | Greedy optimization | May not find global optimum | Use constraint satisfaction with backtracking |

---

## 1. Base Metrics

### 1.1 MAF Heart Rate (Zone 2 Upper Limit)

**Original Formula:**
```
MAF_HR = 180 - Age + Adjustments
```

**Refined Formula:**
```
MAF_HR = base_maf + experience_adj + health_adj + fitness_adj

where:
  base_maf = 180 - Age

  experience_adj = {
    +10  if consistent_training_years >= 4 AND no_injuries_2yr
    +5   if consistent_training_years >= 2
     0   otherwise
  }

  health_adj = {
    -10  if chronic_condition OR recovering_from_illness
    -5   if BMI > 30 OR medications_affecting_hr
     0   otherwise
  }

  fitness_adj = clamp(
    (observed_easy_hr - predicted_easy_hr) / 2,
    -5, +5
  )
```

**Implementation Notes:**
- `observed_easy_hr`: Mean HR from HealthKit for runs at conversational pace (RPE 3-4)
- `predicted_easy_hr`: 60-70% of (220 - Age)
- `fitness_adj` corrects for individual cardiac efficiency

**Validation:**
- Cross-reference with lactate threshold estimates if available
- MAF test: 3-5 mile run at MAF HR should show stable or decreasing pace per mile

### 1.2 Initial Weekly Volume

**Original Formula:**
```
Initial_Volume = 60 min  if no_history or BMI > 30
                 120 min if experienced
```

**Refined Formula:**
```
Initial_Volume = base_volume × history_multiplier × fitness_multiplier × bmi_multiplier

where:
  base_volume = 90 min  # Moderate starting point

  history_multiplier = {
    clamp(recent_4wk_avg / 90, 0.5, 2.0)  if has_history
    0.67                                   if no_history
  }

  fitness_multiplier = {
    1.3   if VO2max_estimate > 50 OR resting_hr < 55
    1.0   if VO2max_estimate 35-50
    0.8   if VO2max_estimate < 35 OR resting_hr > 75
  }

  bmi_multiplier = {
    1.0                          if BMI <= 25
    1.0 - 0.02 × (BMI - 25)     if 25 < BMI <= 35  (caps at 0.8)
    0.8                          if BMI > 35
  }

Final: clamp(Initial_Volume, 30, 300) min/week
```

**Rationale:**
- Continuous scaling prevents abrupt categorization
- History multiplier respects what the runner has been doing
- BMI multiplier is gradual (2% reduction per BMI point above 25)

---

## 2. Load Quantification

### 2.1 TRIMP (Training Impulse)

**Formula (Banister, 1991 - Gender-Specific):**
```
TRIMP = Duration × ΔHR × Y

where:
  ΔHR = (HR_avg - HR_rest) / (HR_max - HR_rest)

  Y_male   = 0.64 × e^(1.92 × ΔHR)
  Y_female = 0.86 × e^(1.67 × ΔHR)

  HR_max = 220 - Age  (or measured)
  HR_rest = from HealthKit (or 60 default)
```

**Implementation:** Already in `core/metrics.py`

### 2.2 EWMA-TRIMP (Exponentially Weighted Moving Average)

**Formula:**
```
EWMA_today = TRIMP_today × λ + (1 - λ) × EWMA_yesterday

where:
  λ_acute  = 2 / (7 + 1)  = 0.25   # 7-day window
  λ_chronic = 2 / (28 + 1) = 0.069  # 28-day window
```

**Implementation:** Already in `core/metrics.py`

### 2.3 ACWR (Acute:Chronic Workload Ratio)

**Formula:**
```
ACWR = Acute_EWMA / Chronic_EWMA

Constraints:
  - Require minimum 14 days chronic history
  - Return None if chronic < 10 (division instability)
```

**Implementation:** Already in `core/metrics.py`

**Critical Note:** Use 7-day LAGGED ACWR for injury prediction (validated in Experiment 002). Concurrent ACWR shows reverse causality.

### 2.4 BMI Stress Modifier (NEW)

**Formula:**
```
stress_modifier = {
  1.0                           if BMI <= 25
  1.0 + 0.03 × (BMI - 25)       if 25 < BMI <= 30  (up to 1.15)
  1.15 + 0.05 × (BMI - 30)      if 30 < BMI <= 40  (up to 1.65)
  1.65                          if BMI > 40
}

Adjusted_TRIMP = TRIMP × stress_modifier
```

**Rationale:**
- Obese runners experience higher cardiovascular stress at same HR
- This modifier inflates TRIMP to reflect true physiological load
- Triggers more conservative Delta V recommendations

---

## 3. Volume Adjustment (Delta V)

### 3.1 Master Delta V Equation

**Validated Optimized Parameters** (from our study):
```
ΔV(ACWR) = {
  CRITICAL (ACWR ≥ 2.0):
    ΔV = -27%

  HIGH RISK (1.5 ≤ ACWR < 2.0):
    ΔV = clamp(-0.26 × (ACWR - 1.3), -16%, -5%)

  CAUTION (1.3 ≤ ACWR < 1.5):
    ΔV = +2%

  OPTIMAL (0.8 ≤ ACWR < 1.3):
    ΔV = clamp(0.30 × (1 - (ACWR - 0.8) / 0.5), +8%, +20%)

  LOW (ACWR < 0.8):
    ΔV = clamp(0.31 × (0.8 - ACWR + 0.1), +14%, +17%)
}

New_Volume = Current_Volume × (1 + ΔV)
```

### 3.2 Beginner/Obese Modifier

**Formula:**
```
max_ΔV_positive = {
  0.10   if experience_level == 'beginner' AND BMI > 30
  0.12   if experience_level == 'beginner' OR BMI > 30
  0.15   if experience_level == 'intermediate'
  0.20   if experience_level == 'advanced'
}

Final_ΔV = clamp(ΔV, -0.30, max_ΔV_positive)
```

### 3.3 Persistent High ACWR Override

**Formula:**
```
if consecutive_weeks_acwr_above_1.5 >= 2:
    ΔV = -30%  # Force major deload
    flag_for_review = True
```

---

## 4. Frequency (Days per Week)

### 4.1 Base Phase Frequency

**Original Formula:**
```
Frequency = min(4, floor(Volume / 20) + 1)
```

**Refined Formula:**
```
base_freq = floor(Volume / 25) + 2

recovery_adj = {
  +1  if recovery_quality >= 0.8 AND sleep_avg >= 7hr
   0  if recovery_quality >= 0.6
  -1  if recovery_quality < 0.5 OR sleep_avg < 6hr
}

Frequency_base = clamp(base_freq + recovery_adj, 2, 4)
```

**Rationale:**
- Minimum 2 days ensures consistency
- Maximum 4 days in base phase allows recovery
- Recovery quality from HRV trends and sleep data

### 4.2 80/20 Phase Frequency

**Formula:**
```
base_freq = floor(Volume / 40) + 3

recovery_adj = {
  +1  if recovery_quality >= 0.8
   0  if recovery_quality >= 0.6
  -1  if recovery_quality < 0.5
}

experience_adj = {
  +1  if experience_level == 'advanced' AND volume > 200 min
   0  otherwise
}

Frequency_80_20 = clamp(base_freq + recovery_adj + experience_adj, 3, 6)
```

### 4.3 Minimum Run Duration Constraint

**Formula:**
```
min_run_duration = max(20, Volume / (Frequency × 1.5))

if any_run_duration < min_run_duration:
    Frequency = Frequency - 1  # Reduce days, increase per-run duration
```

---

## 5. Workout Types and Splits

### 5.1 Training Phases

**Phase Definitions:**
```
Phase = {
  'base_building':   weeks 1-8,  100% Zone 2
  'transition':      weeks 9-10, 90% easy / 10% tempo
  '80_20':           weeks 11+,  80% easy / 20% hard
}
```

### 5.2 Base Phase Splits

**Formula:**
```
For Frequency F days and Volume V minutes:

long_run_duration = 0.30 × V  # 30% of volume in long run
short_run_duration = (0.70 × V) / (F - 1)  # Remaining across other days

Constraints:
  - long_run_duration <= 0.35 × V  # Never more than 35%
  - long_run_duration >= 30 min
  - short_run_duration >= 20 min

Workout_types = {
  'easy_short': short_run_duration, count = F - 1
  'easy_long':  long_run_duration,  count = 1
}

All runs at MAF HR (Zone 2)
```

### 5.3 80/20 Phase Splits

**Formula:**
```
For Frequency F days and Volume V minutes:

easy_volume = 0.80 × V
hard_volume = 0.20 × V

long_run_duration = 0.25 × V  # 25% in long run
hard_session_duration = hard_volume  # Can be 1-2 sessions

if F >= 5:
    hard_sessions = 2
    hard_per_session = hard_volume / 2
else:
    hard_sessions = 1
    hard_per_session = hard_volume

easy_short_duration = (easy_volume - long_run_duration) / (F - 1 - hard_sessions)

Workout_types = {
  'easy_short':  easy_short_duration, count = F - 1 - hard_sessions
  'easy_long':   long_run_duration,   count = 1
  'hard_tempo':  hard_per_session,    count = hard_sessions (if applicable)
  'hard_interval': hard_per_session,  count = hard_sessions (alternating)
}
```

### 5.4 Hard Workout Intensity Zones

**Definitions:**
```
Tempo Run:
  - HR: 83-87% of HR_max
  - Duration: 20-40 min continuous
  - RPE: 6-7

Interval Run:
  - Work HR: 90-95% of HR_max
  - Rest HR: < 70% of HR_max
  - Structure: 4-6 × (3-5 min work / 2-3 min recovery)
  - RPE: 8-9 during work

Threshold Run:
  - HR: 88-92% of HR_max
  - Duration: 15-25 min continuous
  - RPE: 7-8
```

---

## 6. Spacing and Order

### 6.1 Consecutive Run Limit

**Formula:**
```
max_consecutive = {
  2   if experience_level == 'beginner' OR BMI > 30
  2   if Volume < 120 min
  3   if Volume 120-200 min AND experience_level >= 'intermediate'
  4   if Volume > 200 min AND experience_level == 'advanced'
}

Constraint: No more than max_consecutive running days in a row
```

### 6.2 Day Stress Scoring

**Formula:**
```
stress_score(workout) = {
  1.0  for easy_short
  2.0  for easy_long
  2.5  for hard_tempo
  3.0  for hard_interval
}

proximity_penalty(day, schedule) = {
  +2.0  if adjacent day has stress_score >= 2.5
  +1.0  if adjacent day has stress_score >= 2.0
   0    otherwise
}
```

### 6.3 Day Assignment Algorithm

**Constraint Satisfaction Approach:**
```
def assign_days(workouts, available_days, max_consecutive):
    # Sort workouts by stress (highest first)
    workouts = sorted(workouts, key=stress_score, reverse=True)

    schedule = {}

    for workout in workouts:
        best_day = None
        best_score = infinity

        for day in available_days:
            if violates_consecutive(day, schedule, max_consecutive):
                continue
            if violates_hard_spacing(day, workout, schedule):
                continue

            score = proximity_penalty(day, schedule)

            # Prefer weekend for long runs
            if workout.type == 'easy_long' and day in [Saturday, Sunday]:
                score -= 0.5

            # Prefer midweek for hard workouts
            if workout.type.startswith('hard') and day in [Tuesday, Wednesday, Thursday]:
                score -= 0.3

            if score < best_score:
                best_score = score
                best_day = day

        schedule[best_day] = workout
        available_days.remove(best_day)

    return schedule
```

### 6.4 Hard Workout Spacing

**Constraint:**
```
min_days_between_hard = {
  2  if experience_level == 'beginner'
  2  if Frequency <= 4
  1  if experience_level == 'advanced' AND Frequency >= 5
}

# Never schedule hard workouts on consecutive days
# At least min_days_between_hard easy/rest days between hard sessions
```

---

## 7. Phase Transition Logic

### 7.1 Base → 80/20 Readiness

**Criteria:**
```
ready_for_80_20 = (
    weeks_in_base >= 6 AND
    ACWR in [0.8, 1.2] for last 2 weeks AND
    maf_pace_improvement >= 0 AND  # Not getting slower
    no_injuries_last_4_weeks AND
    volume >= 90 min/week
)
```

### 7.2 Transition Phase

**Formula:**
```
During weeks 9-10 (transition):
  hard_percentage = week_number == 9 ? 0.10 : 0.15
  easy_percentage = 1.0 - hard_percentage

  Hard workout = tempo run only (no intervals yet)
```

### 7.3 Deload Weeks

**Formula:**
```
deload_frequency = every 4th week (or if ACWR > 1.3 for 2+ weeks)

deload_volume = current_volume × 0.60
deload_intensity = easy runs only (drop hard sessions)
deload_frequency = current_frequency - 1
```

---

## 8. Integration: The RunPlanEngine

### 8.1 Weekly Update Cycle

```
def weekly_update(athlete, current_plan, healthkit_data):
    # 1. Calculate load metrics
    trimp_history = calculate_all_trimp(healthkit_data)
    acwr = calculate_acwr(trimp_history)

    # 2. Apply BMI modifier if applicable
    if athlete.bmi > 25:
        acwr = adjust_acwr_for_bmi(acwr, athlete.bmi)

    # 3. Calculate Delta V
    delta_v, zone, flag = calculate_delta_v(acwr, athlete.experience_level)

    # 4. Apply beginner/obese cap
    delta_v = apply_experience_cap(delta_v, athlete)

    # 5. Calculate new volume
    new_volume = current_plan.volume * (1 + delta_v)

    # 6. Determine frequency
    new_frequency = calculate_frequency(new_volume, athlete, current_plan.phase)

    # 7. Determine workout splits
    workouts = calculate_splits(new_volume, new_frequency, current_plan.phase)

    # 8. Assign days
    schedule = assign_days(workouts, athlete.available_days, athlete.max_consecutive)

    # 9. Check phase transition
    if should_transition(athlete, current_plan):
        current_plan.phase = next_phase(current_plan.phase)

    # 10. Check deload
    if should_deload(athlete, current_plan):
        schedule = apply_deload(schedule)

    return new_plan(
        volume=new_volume,
        frequency=new_frequency,
        workouts=workouts,
        schedule=schedule,
        phase=current_plan.phase
    )
```

---

## 9. Parameter Configuration

### 9.1 Tunable Parameters

```json
{
  "maf": {
    "base_formula": 180,
    "experience_bonus_4yr": 10,
    "experience_bonus_2yr": 5,
    "health_penalty_chronic": -10,
    "health_penalty_bmi30": -5
  },
  "volume": {
    "initial_base": 90,
    "min_volume": 30,
    "max_volume": 300,
    "bmi_penalty_per_point": 0.02,
    "bmi_penalty_start": 25
  },
  "delta_v": {
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
  },
  "frequency": {
    "base_phase_min": 2,
    "base_phase_max": 4,
    "eighty_twenty_min": 3,
    "eighty_twenty_max": 6,
    "volume_per_day_base": 25,
    "volume_per_day_8020": 40
  },
  "splits": {
    "long_run_base_pct": 0.30,
    "long_run_max_pct": 0.35,
    "easy_pct_8020": 0.80,
    "hard_pct_8020": 0.20
  },
  "spacing": {
    "beginner_max_consecutive": 2,
    "intermediate_max_consecutive": 3,
    "advanced_max_consecutive": 4,
    "min_days_between_hard": 2
  },
  "phases": {
    "min_base_weeks": 6,
    "transition_weeks": 2,
    "deload_frequency_weeks": 4,
    "deload_volume_multiplier": 0.60
  }
}
```

---

## 10. Validation Plan

### 10.1 Equation-Level Testing

| Equation | Test Method | Success Criteria |
|----------|-------------|------------------|
| MAF HR | Compare to lactate threshold data | Within ±5 bpm of LT1 |
| Initial Volume | Backtest on training histories | No >1.5 ACWR in week 2 |
| Delta V | Bootstrap simulation | <10% high-risk events |
| Frequency | User adherence tracking | >80% completion rate |
| Splits | Compare to Fitzgerald plans | Distribution matches 80/20 |
| Spacing | Injury correlation | No injuries day after hard |

### 10.2 Integration Testing

1. **Synthetic Runner Simulation**: 1000 athletes, 12 weeks
   - Target: Mean volume growth 2.0x with <15% injury rate

2. **Historical Backtest**: Apply to PMData/Runners datasets
   - Target: Delta V would have prevented >50% of high-ACWR injuries

3. **A/B Testing**: Conservative vs optimized parameters
   - Target: Optimized shows +5% growth with equivalent safety

---

## 11. Implementation Roadmap

### Phase 1: Core Equations (This PR)
- [x] MAF HR calculation
- [x] Initial Volume calculation
- [x] BMI stress modifier
- [x] Frequency equations
- [x] Workout splits
- [x] Day assignment algorithm

### Phase 2: Integration
- [ ] RunPlanEngine class
- [ ] HealthKit data adapter
- [ ] Weekly update cycle

### Phase 3: Validation
- [ ] Unit tests for each equation
- [ ] Integration tests with synthetic data
- [ ] Backtest on real datasets

### Phase 4: Refinement
- [ ] Parameter optimization via Bayesian search
- [ ] A/B testing infrastructure
- [ ] User feedback integration

---

## Appendix A: Key Literature References

1. Maffetone, P. (2010). *The Big Book of Endurance Training and Racing*
2. Fitzgerald, M. (2014). *80/20 Running*
3. Banister, E.W. (1991). TRIMP formula development
4. Williams, S. et al. (2017). ACWR and injury risk
5. Gabbett, T.J. (2016). Training-injury prevention paradox
6. Seiler, S. (2010). Polarized training distribution

---

## Appendix B: HealthKit Data Requirements

```swift
// Required data types
HKQuantityType.heartRate
HKQuantityType.restingHeartRate
HKQuantityType.vo2Max
HKQuantityType.bodyMass
HKQuantityType.height
HKWorkoutType.running

// Optional enhancements
HKQuantityType.heartRateVariabilitySDNN
HKCategoryType.sleepAnalysis
```
