"""
PMData Loader for Delta V Validation
=====================================

Loads and processes the PMData dataset (Simula Research Lab) for
ACWR (Acute:Chronic Workload Ratio) analysis and injury prediction.

The dataset contains:
- 16 participants over ~5 months
- Fitbit HR data (continuous)
- sRPE (session Rating of Perceived Exertion) - training load
- Wellness questionnaires (fatigue, mood, sleep, soreness)
- Injury reports with body part and severity

Key approach:
- Use sRPE × duration as training load (Foster's method)
- Calculate ACWR using 7-day acute / 28-day chronic EWMA
- Correlate with actual injury events

Author: Delta V Backtesting Project
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import ast
import warnings


@dataclass
class InjuryEvent:
    """Represents an injury event."""
    participant_id: str
    date: datetime
    body_parts: Dict[str, str]  # e.g., {'left_knee': 'minor'}


@dataclass
class TrainingSession:
    """Represents a training session."""
    participant_id: str
    date: datetime
    activity_types: List[str]
    rpe: int  # Rate of Perceived Exertion (1-10 or Borg 6-20)
    duration_min: int
    load: float  # sRPE = RPE × duration


class PMDataLoader:
    """
    Data loader for PMData (Simula) dataset.

    Provides methods to:
    - Load sRPE training data and calculate daily loads
    - Load injury events
    - Calculate ACWR using EWMA
    - Analyze ACWR-injury relationships
    """

    def __init__(self, data_dir: str = None):
        """
        Initialize the loader.

        Args:
            data_dir: Path to pmdata directory
        """
        if data_dir is None:
            data_dir = Path(__file__).parent / "pmdata"
        else:
            data_dir = Path(data_dir)

        self.data_dir = data_dir
        self.participants: List[str] = []
        self.training_data: Dict[str, pd.DataFrame] = {}
        self.injury_events: List[InjuryEvent] = []
        self.wellness_data: Dict[str, pd.DataFrame] = {}
        self._loaded = False

        # ACWR parameters
        self.acute_window = 7
        self.chronic_window = 28

    def _discover_participants(self) -> List[str]:
        """Find all participant directories."""
        participants = []
        for p in sorted(self.data_dir.iterdir()):
            if p.is_dir() and p.name.startswith('p'):
                participants.append(p.name)
        return participants

    def load_all(self, verbose: bool = True) -> 'PMDataLoader':
        """
        Load all data from PMData.

        Args:
            verbose: Print progress

        Returns:
            self for chaining
        """
        if verbose:
            print("Loading PMData Dataset...")

        self.participants = self._discover_participants()
        if verbose:
            print(f"  Found {len(self.participants)} participants")

        # Load training (sRPE) data
        total_sessions = 0
        for pid in self.participants:
            df = self._load_srpe(pid)
            if df is not None and len(df) > 0:
                self.training_data[pid] = df
                total_sessions += len(df)

        if verbose:
            print(f"  Training sessions: {total_sessions:,} across {len(self.training_data)} participants")

        # Load injury events
        self.injury_events = self._load_all_injuries()
        if verbose:
            print(f"  Injury events: {len(self.injury_events)}")

        # Load wellness data
        total_wellness = 0
        for pid in self.participants:
            df = self._load_wellness(pid)
            if df is not None and len(df) > 0:
                self.wellness_data[pid] = df
                total_wellness += len(df)

        if verbose:
            print(f"  Wellness records: {total_wellness:,}")

        self._loaded = True
        return self

    def _load_srpe(self, participant_id: str) -> Optional[pd.DataFrame]:
        """Load sRPE data for a participant."""
        srpe_path = self.data_dir / participant_id / "pmsys" / "srpe.csv"

        if not srpe_path.exists():
            return None

        try:
            df = pd.read_csv(srpe_path)
            df['datetime'] = pd.to_datetime(df['end_date_time'])
            df['date'] = df['datetime'].dt.date

            # Calculate sRPE load (RPE × duration)
            df['load'] = df['perceived_exertion'] * df['duration_min']

            # Parse activity types
            df['activities'] = df['activity_names'].apply(
                lambda x: ast.literal_eval(x) if pd.notna(x) else []
            )

            # Check if it's running-related
            df['is_running'] = df['activities'].apply(
                lambda x: 'running' in x or 'endurance' in x
            )

            df['participant_id'] = participant_id

            return df[['participant_id', 'date', 'datetime', 'activities',
                       'perceived_exertion', 'duration_min', 'load', 'is_running']]
        except Exception as e:
            print(f"  Warning: Error loading sRPE for {participant_id}: {e}")
            return None

    def _load_wellness(self, participant_id: str) -> Optional[pd.DataFrame]:
        """Load wellness data for a participant."""
        wellness_path = self.data_dir / participant_id / "pmsys" / "wellness.csv"

        if not wellness_path.exists():
            return None

        try:
            df = pd.read_csv(wellness_path)
            df['datetime'] = pd.to_datetime(df['effective_time_frame'])
            df['date'] = df['datetime'].dt.date
            df['participant_id'] = participant_id

            return df
        except Exception as e:
            print(f"  Warning: Error loading wellness for {participant_id}: {e}")
            return None

    def _load_all_injuries(self) -> List[InjuryEvent]:
        """Load all injury events from all participants."""
        injuries = []

        for pid in self.participants:
            injury_path = self.data_dir / pid / "pmsys" / "injury.csv"

            if not injury_path.exists():
                continue

            try:
                df = pd.read_csv(injury_path)

                for _, row in df.iterrows():
                    injury_str = row['injuries']

                    # Skip empty injuries
                    if injury_str == '{}' or pd.isna(injury_str):
                        continue

                    try:
                        # Parse the injury dict
                        body_parts = ast.literal_eval(injury_str)

                        if body_parts:  # Non-empty dict
                            injuries.append(InjuryEvent(
                                participant_id=pid,
                                date=pd.to_datetime(row['effective_time_frame']),
                                body_parts=body_parts
                            ))
                    except:
                        continue

            except Exception as e:
                print(f"  Warning: Error loading injuries for {pid}: {e}")

        return injuries

    def get_daily_load(self, participant_id: str) -> pd.DataFrame:
        """
        Get daily training load for a participant.

        Aggregates multiple sessions on the same day.

        Args:
            participant_id: The participant ID (e.g., 'p01')

        Returns:
            DataFrame with date, total_load, session_count, is_running
        """
        if participant_id not in self.training_data:
            return pd.DataFrame()

        df = self.training_data[participant_id].copy()

        daily = df.groupby('date').agg({
            'load': 'sum',
            'duration_min': 'sum',
            'perceived_exertion': 'mean',
            'is_running': 'any'
        }).reset_index()

        daily.columns = ['date', 'total_load', 'total_duration', 'avg_rpe', 'has_running']
        daily['session_count'] = df.groupby('date').size().values

        return daily

    def _compute_ewma(self, values: np.ndarray, window: int) -> np.ndarray:
        """Calculate exponentially weighted moving average."""
        alpha = 2 / (window + 1)
        ewma = np.zeros(len(values))
        ewma[0] = values[0]

        for i in range(1, len(values)):
            ewma[i] = alpha * values[i] + (1 - alpha) * ewma[i-1]

        return ewma

    def get_acwr_series(self, participant_id: str,
                        min_days: int = 21,
                        fill_gaps: bool = True) -> pd.DataFrame:
        """
        Calculate ACWR time series for a participant.

        Args:
            participant_id: The participant ID
            min_days: Minimum days before ACWR is valid
            fill_gaps: Fill missing days with 0 load

        Returns:
            DataFrame with date, load, acute, chronic, acwr
        """
        daily = self.get_daily_load(participant_id)

        if len(daily) == 0:
            return pd.DataFrame()

        daily['date'] = pd.to_datetime(daily['date'])

        if fill_gaps:
            # Create complete date range
            date_range = pd.date_range(
                start=daily['date'].min(),
                end=daily['date'].max(),
                freq='D'
            )

            # Reindex to fill gaps
            daily = daily.set_index('date').reindex(date_range).reset_index()
            daily.columns = ['date'] + list(daily.columns[1:])

            # Fill missing load with 0 (rest day)
            daily['total_load'] = daily['total_load'].fillna(0)
            daily['total_duration'] = daily['total_duration'].fillna(0)
            daily['session_count'] = daily['session_count'].fillna(0)

        # Calculate EWMA
        loads = daily['total_load'].values
        acute = self._compute_ewma(loads, self.acute_window)
        chronic = self._compute_ewma(loads, self.chronic_window)

        # Calculate ACWR
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            acwr = np.where(chronic > 1e-6, acute / chronic, np.nan)

        result = pd.DataFrame({
            'date': daily['date'],
            'load': loads,
            'acute_load': acute,
            'chronic_load': chronic,
            'acwr': acwr,
            'duration': daily['total_duration'],
            'sessions': daily['session_count']
        })

        # Mark early days as NaN
        result.loc[:min_days-1, 'acwr'] = np.nan

        return result

    def get_injury_dates(self, participant_id: str = None) -> pd.DataFrame:
        """
        Get injury dates with context.

        Args:
            participant_id: Optional filter by participant

        Returns:
            DataFrame with injury dates and ACWR at time of injury
        """
        injuries = self.injury_events

        if participant_id:
            injuries = [i for i in injuries if i.participant_id == participant_id]

        records = []
        for injury in injuries:
            pid = injury.participant_id

            # Get ACWR at time of injury
            acwr_data = self.get_acwr_series(pid)

            if len(acwr_data) == 0:
                continue

            injury_date = injury.date.date() if hasattr(injury.date, 'date') else injury.date

            # Find closest ACWR value
            acwr_data['date_only'] = acwr_data['date'].dt.date
            match = acwr_data[acwr_data['date_only'] == injury_date]

            if len(match) > 0:
                row = match.iloc[0]
                acwr_val = row['acwr']
                load_val = row['load']
                acute_val = row['acute_load']
                chronic_val = row['chronic_load']
            else:
                acwr_val = np.nan
                load_val = np.nan
                acute_val = np.nan
                chronic_val = np.nan

            records.append({
                'participant_id': pid,
                'date': injury.date,
                'body_parts': injury.body_parts,
                'severity': list(injury.body_parts.values())[0] if injury.body_parts else None,
                'acwr': acwr_val,
                'load': load_val,
                'acute_load': acute_val,
                'chronic_load': chronic_val
            })

        return pd.DataFrame(records)

    def compute_all_acwr(self, min_days: int = 21) -> pd.DataFrame:
        """
        Calculate ACWR for all participants.

        Returns:
            Combined DataFrame with all participants' ACWR data
        """
        all_data = []

        for pid in self.training_data.keys():
            acwr_data = self.get_acwr_series(pid, min_days)
            if len(acwr_data) > 0:
                acwr_data['participant_id'] = pid
                all_data.append(acwr_data)

        if not all_data:
            return pd.DataFrame()

        return pd.concat(all_data, ignore_index=True)

    def analyze_injury_acwr(self) -> Dict:
        """
        Analyze ACWR values at time of injury vs non-injury.

        Returns:
            Dictionary with analysis results
        """
        injury_df = self.get_injury_dates()
        all_acwr = self.compute_all_acwr()

        # Get ACWR values at injury times
        injury_acwr = injury_df['acwr'].dropna()

        # Get all non-injury ACWR values
        valid_acwr = all_acwr.dropna(subset=['acwr'])

        # Mark injury days
        injury_dates = set()
        for _, row in injury_df.iterrows():
            injury_dates.add((row['participant_id'], row['date'].date() if hasattr(row['date'], 'date') else row['date']))

        valid_acwr['is_injury'] = valid_acwr.apply(
            lambda x: (x['participant_id'], x['date'].date() if hasattr(x['date'], 'date') else x['date']) in injury_dates,
            axis=1
        )

        non_injury_acwr = valid_acwr[~valid_acwr['is_injury']]['acwr']

        # Calculate statistics
        result = {
            'n_injuries': len(injury_acwr),
            'n_non_injury_days': len(non_injury_acwr),
            'injury_acwr_mean': injury_acwr.mean() if len(injury_acwr) > 0 else np.nan,
            'injury_acwr_std': injury_acwr.std() if len(injury_acwr) > 0 else np.nan,
            'injury_acwr_median': injury_acwr.median() if len(injury_acwr) > 0 else np.nan,
            'non_injury_acwr_mean': non_injury_acwr.mean(),
            'non_injury_acwr_std': non_injury_acwr.std(),
            'non_injury_acwr_median': non_injury_acwr.median(),
        }

        # Calculate percentage of injuries at high ACWR
        if len(injury_acwr) > 0:
            result['pct_injuries_acwr_gt_1_3'] = (injury_acwr > 1.3).sum() / len(injury_acwr) * 100
            result['pct_injuries_acwr_gt_1_5'] = (injury_acwr > 1.5).sum() / len(injury_acwr) * 100
        else:
            result['pct_injuries_acwr_gt_1_3'] = np.nan
            result['pct_injuries_acwr_gt_1_5'] = np.nan

        # Calculate injury rates by ACWR zone
        zones = [
            (0, 0.8, 'low'),
            (0.8, 1.3, 'optimal'),
            (1.3, 1.5, 'caution'),
            (1.5, float('inf'), 'high')
        ]

        zone_stats = []
        for low, high, name in zones:
            zone_data = valid_acwr[(valid_acwr['acwr'] >= low) & (valid_acwr['acwr'] < high)]
            n_days = len(zone_data)
            n_injuries = zone_data['is_injury'].sum()

            zone_stats.append({
                'zone': name,
                'acwr_range': f"{low}-{high if high != float('inf') else '+'}",
                'n_days': n_days,
                'n_injuries': n_injuries,
                'injury_rate_per_1000': (n_injuries / n_days * 1000) if n_days > 0 else 0
            })

        result['zone_analysis'] = pd.DataFrame(zone_stats)

        return result

    def get_week_over_week_changes(self, participant_id: str) -> pd.DataFrame:
        """
        Calculate week-over-week load changes (like Delta V would prescribe).

        Args:
            participant_id: The participant ID

        Returns:
            DataFrame with weekly loads and changes
        """
        acwr_data = self.get_acwr_series(participant_id)

        if len(acwr_data) == 0:
            return pd.DataFrame()

        # Group by week
        acwr_data['week'] = acwr_data['date'].dt.isocalendar().week
        acwr_data['year'] = acwr_data['date'].dt.year

        weekly = acwr_data.groupby(['year', 'week']).agg({
            'load': 'sum',
            'acwr': 'last',  # End-of-week ACWR
            'date': 'last'
        }).reset_index()

        # Calculate week-over-week change
        weekly['prev_load'] = weekly['load'].shift(1)
        weekly['load_change_pct'] = (weekly['load'] - weekly['prev_load']) / weekly['prev_load'] * 100

        return weekly

    def validate_delta_v_zones(self) -> Dict:
        """
        Validate the Delta V zone definitions against real injury data.

        Tests if the zone boundaries (0.8, 1.3, 1.5, 2.0) are appropriate.

        Returns:
            Dictionary with validation results
        """
        all_acwr = self.compute_all_acwr()
        injury_df = self.get_injury_dates()

        # Mark injury days
        injury_dates = set()
        for _, row in injury_df.iterrows():
            pid = row['participant_id']
            date = row['date'].date() if hasattr(row['date'], 'date') else row['date']
            injury_dates.add((pid, date))

        all_acwr['is_injury'] = all_acwr.apply(
            lambda x: (x['participant_id'], x['date'].date()) in injury_dates,
            axis=1
        )

        valid_data = all_acwr.dropna(subset=['acwr'])

        # Test different threshold values
        thresholds = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

        results = []
        for thresh in thresholds:
            above = valid_data[valid_data['acwr'] > thresh]
            below = valid_data[valid_data['acwr'] <= thresh]

            inj_above = above['is_injury'].sum()
            inj_below = below['is_injury'].sum()

            rate_above = inj_above / len(above) * 1000 if len(above) > 0 else 0
            rate_below = inj_below / len(below) * 1000 if len(below) > 0 else 0

            results.append({
                'threshold': thresh,
                'n_days_above': len(above),
                'n_days_below': len(below),
                'injuries_above': inj_above,
                'injuries_below': inj_below,
                'rate_above': rate_above,
                'rate_below': rate_below,
                'relative_risk': rate_above / rate_below if rate_below > 0 else np.nan
            })

        return {
            'threshold_analysis': pd.DataFrame(results),
            'total_injuries': len(injury_df),
            'total_days': len(valid_data),
            'overall_injury_rate': len(injury_df) / len(valid_data) * 1000 if len(valid_data) > 0 else 0
        }


def run_validation():
    """
    Run full validation on PMData.
    """
    print("=" * 70)
    print("PMDATA DATASET - ACWR INJURY VALIDATION")
    print("=" * 70)
    print()

    # Load data
    loader = PMDataLoader()
    loader.load_all()

    # Analyze ACWR-injury relationship
    print("\n" + "-" * 70)
    print("ACWR-INJURY ANALYSIS")
    print("-" * 70)

    analysis = loader.analyze_injury_acwr()

    print(f"\nSample Sizes:")
    print(f"  Injury events: {analysis['n_injuries']}")
    print(f"  Non-injury days: {analysis['n_non_injury_days']:,}")

    print(f"\nACWR Statistics:")
    print(f"  Injury days - Mean ACWR: {analysis['injury_acwr_mean']:.3f} (SD: {analysis['injury_acwr_std']:.3f})")
    print(f"  Non-injury days - Mean ACWR: {analysis['non_injury_acwr_mean']:.3f} (SD: {analysis['non_injury_acwr_std']:.3f})")

    diff = analysis['injury_acwr_mean'] - analysis['non_injury_acwr_mean']
    print(f"\n  ACWR Difference (injury - non-injury): {diff:+.3f}")

    print(f"\nPercentage of injuries at high ACWR:")
    print(f"  ACWR > 1.3: {analysis['pct_injuries_acwr_gt_1_3']:.1f}%")
    print(f"  ACWR > 1.5: {analysis['pct_injuries_acwr_gt_1_5']:.1f}%")

    print("\n" + "-" * 70)
    print("INJURY RATES BY ACWR ZONE")
    print("-" * 70)

    zone_df = analysis['zone_analysis']
    print(f"\n  {'Zone':<10} {'ACWR Range':<12} {'Days':>8} {'Injuries':>10} {'Rate/1000':>12}")
    print(f"  {'-'*52}")

    for _, row in zone_df.iterrows():
        print(f"  {row['zone']:<10} {row['acwr_range']:<12} {row['n_days']:>8} {row['n_injuries']:>10} {row['injury_rate_per_1000']:>12.2f}")

    # Validate thresholds
    print("\n" + "-" * 70)
    print("THRESHOLD VALIDATION")
    print("-" * 70)

    validation = loader.validate_delta_v_zones()
    thresh_df = validation['threshold_analysis']

    print(f"\n  {'Threshold':>10} {'Days Above':>12} {'Inj Above':>10} {'Rate Above':>12} {'Rel Risk':>10}")
    print(f"  {'-'*54}")

    for _, row in thresh_df.iterrows():
        rr_str = f"{row['relative_risk']:.2f}" if not np.isnan(row['relative_risk']) else "N/A"
        print(f"  {row['threshold']:>10.1f} {row['n_days_above']:>12} {row['injuries_above']:>10} {row['rate_above']:>12.2f} {rr_str:>10}")

    # Show individual injury events with ACWR
    print("\n" + "-" * 70)
    print("INDIVIDUAL INJURY EVENTS WITH ACWR")
    print("-" * 70)

    injury_df = loader.get_injury_dates()
    print(f"\n  {'Participant':<12} {'Date':<12} {'ACWR':>8} {'Load':>8} {'Body Parts'}")
    print(f"  {'-'*60}")

    for _, row in injury_df.iterrows():
        acwr_str = f"{row['acwr']:.2f}" if not np.isnan(row['acwr']) else "N/A"
        load_str = f"{row['load']:.0f}" if not np.isnan(row['load']) else "N/A"
        date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])[:10]
        parts = ', '.join(row['body_parts'].keys()) if row['body_parts'] else 'N/A'
        print(f"  {row['participant_id']:<12} {date_str:<12} {acwr_str:>8} {load_str:>8} {parts}")

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY FOR DELTA-V EQUATION")
    print("=" * 70)

    # Find optimal threshold
    valid_rr = thresh_df[thresh_df['relative_risk'].notna()]
    if len(valid_rr) > 0:
        best_thresh = valid_rr.loc[valid_rr['relative_risk'].idxmax()]
        print(f"\nBest differentiating threshold: ACWR > {best_thresh['threshold']}")
        print(f"  Relative risk at this threshold: {best_thresh['relative_risk']:.2f}x")

    # Key findings
    high_zone = zone_df[zone_df['zone'] == 'high']
    optimal_zone = zone_df[zone_df['zone'] == 'optimal']

    if len(high_zone) > 0 and len(optimal_zone) > 0:
        high_rate = high_zone.iloc[0]['injury_rate_per_1000']
        optimal_rate = optimal_zone.iloc[0]['injury_rate_per_1000']

        if optimal_rate > 0:
            rr = high_rate / optimal_rate
            print(f"\nHigh ACWR (>1.5) relative risk vs optimal (0.8-1.3): {rr:.2f}x")

    print("\n" + "=" * 70)

    return loader


if __name__ == "__main__":
    loader = run_validation()
