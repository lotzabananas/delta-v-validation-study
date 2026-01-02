# Delta V Project Status (Pre-Compaction Snapshot)

## What We Built
- Complete backtesting framework for Delta V running volume equation
- TRIMP, EWMA, ACWR calculations in `core/metrics.py`
- Parameterized Delta V equation in `core/delta_v.py`
- Simulation engine in `simulation/engine.py`
- Bayesian optimization in `optimization/search.py`
- Jupyter notebook for analysis in `notebooks/delta_v_analysis.ipynb`

## Key Results
- **Baseline**: 1.82x growth, 30% target reached, 3.8% risk rate
- **Optimized**: 2.50x growth, 55% target reached, 5.0% risk rate
- Optimized params saved in `optimized_params.json`

## Data Downloaded
1. **Kaggle Running HR** - `data/kaggle/` ✅ (lap summaries, HR data)
2. **Zenodo Triathlete** - `data/zenodo_triathlete/` ✅
   - `athletes.csv` (1000 athletes)
   - `daily_data.csv` (366K records with **injury labels**)
   - `activity_data.csv` (training sessions)
3. **PMData** - User downloading, will place in `data/pmdata/`

## Critical Files
- `core/delta_v.py` - The equation
- `core/metrics.py` - TRIMP/ACWR
- `simulation/engine.py` - Backtesting
- `optimized_params.json` - Best parameters
- `data/kaggle_loader.py` - Kaggle data loader (created)

## Next Steps
1. Build `data/triathlete_loader.py` for Zenodo data
2. Validate Delta V against real injury outcomes
3. Calculate if high ACWR correlates with injury=1 in Zenodo data
4. Test optimized parameters on real data

## Key Insight from Devil's Advocate
The synthetic validation is circular - we need REAL injury data to validate.
The Zenodo dataset has injury labels - this is the real test!

## Commands to Run Framework
```bash
cd "/Users/timmac/Desktop/Delta V backtesting"
source venv/bin/activate
python main.py test      # Run tests
python main.py backtest  # Run baseline
python main.py optimize --trials 100  # Optimize
jupyter notebook notebooks/delta_v_analysis.ipynb  # Interactive
```
