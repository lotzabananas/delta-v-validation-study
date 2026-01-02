# Data Acquisition Plan for Delta V Validation

## Status: Kaggle Data Downloaded ✅

Unzipped to: `/Users/timmac/Desktop/Delta V backtesting/data/kaggle/`

---

## Datasets to Download

### 1. PMData (Simula) - ~750 MB - PRIORITY
**Best for injury validation - has real injury reports!**

**STATUS: User downloading manually - place contents in `data/pmdata/`**

Contents: 16 participants, 5 months, HR data, sRPE, injuries, wellness

Expected structure after extraction:
```
data/pmdata/
├── participant_1/
│   ├── activity/
│   ├── heartrate/
│   └── ...
├── participant_2/
...
```

---

### 2. Synthetic Triathlete (Zenodo) - 187 MB
**1000 athletes with injury labels for statistical power**

```bash
cd /Users/timmac/Desktop/Delta\ V\ backtesting/data
mkdir -p zenodo_triathlete
curl -L -o zenodo_triathlete/athletes.csv "https://zenodo.org/records/15401061/files/athletes.csv?download=1"
curl -L -o zenodo_triathlete/daily_data.csv "https://zenodo.org/records/15401061/files/daily_data.csv?download=1"
curl -L -o zenodo_triathlete/activity_data.csv "https://zenodo.org/records/15401061/files/activity_data.csv?download=1"
```

Contents: 366,000 daily records, HR, HRV, sleep, injury labels

---

### 3. Long-Distance Running (Figshare) - ~2 GB
**36,000 real runners for volume progression patterns**

```bash
cd /Users/timmac/Desktop/Delta\ V\ backtesting/data
mkdir -p figshare_running
curl -L -o figshare_running/running_data.zip "https://figshare.com/ndownloader/files/30570599"
unzip figshare_running/running_data.zip -d figshare_running/
```

Contents: 10M sessions, distance, duration (no HR - use for volume patterns only)

---

## NOT Downloading
- **SoccerMon**: 93 GB - too large
- **PhysioNet datasets**: Require credentialed access

---

## After Downloads - Create Loaders

Each dataset needs a loader in `/data/`:
1. `pmdata_loader.py` - Parse HR JSON, calculate TRIMP, load injuries
2. `triathlete_loader.py` - Load daily data, extract ACWR, injury correlation
3. `figshare_loader.py` - Load running sessions, analyze volume patterns

---

## Validation Plan

1. **PMData**: Test if our optimized Delta V parameters correlate with actual injury events
2. **Triathlete**: Run 1000-athlete simulation, compare injury predictions to labels
3. **Figshare**: Validate volume progression patterns against real runners

---

## Project Summary (for context recovery)

- Built Delta V equation for weekly running volume adjustments based on ACWR
- Created backtesting framework with synthetic runner profiles
- Optimized parameters: 37% better growth, 55% target achievement (vs 30% baseline)
- Key files:
  - `core/delta_v.py` - The equation
  - `core/metrics.py` - TRIMP, EWMA, ACWR
  - `simulation/engine.py` - Backtesting
  - `optimization/search.py` - Bayesian optimization
  - `optimized_params.json` - Best parameters found
  - `notebooks/delta_v_analysis.ipynb` - Interactive analysis
