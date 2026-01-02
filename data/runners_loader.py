"""
Mid-Long Distance Runners Dataset Loader
=========================================

Loads and processes the Mid-Long Distance Runners dataset from timeseries (daily).csv.
This dataset contains daily training data for 74 athletes with injury labels.

Dataset source: https://www.kaggle.com/datasets
Structure: Each row contains current day + 6 days of lagged features

Key columns per day:
- nr. sessions: Number of training sessions
- total km: Total kilometers run
- km Z3-4: Kilometers in zone 3-4 (threshold/tempo)
- km Z5-T1-T2: Kilometers in zone 5 and T1-T2 (VO2max/intervals)
- km sprinting: Sprint kilometers
- strength training: Strength training indicator
- hours alternative: Hours of cross-training
- perceived exertion: RPE score
- perceived trainingSuccess: Self-rated training success
- perceived recovery: Self-rated recovery

Target: injury (0/1)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


@dataclass
class RunnersDatasetStats:
    """Statistics about the loaded dataset."""
    n_athletes: int
    n_days: int
    n_injuries: int
    injury_rate: float
    date_range: Tuple[int, int]
    avg_days_per_athlete: float


class RunnersDataLoader:
    """
    Loader for Mid-Long Distance Runners dataset.

    Calculates ACWR from daily training load (total km).
    """

    def __init__(self, data_path: str = None):
        """
        Initialize the loader.

        Args:
            data_path: Path to the CSV file (uses default if None)
        """
        if data_path is None:
            base = Path(__file__).parent.parent
            data_path = base / "New Data from Tim" / "timeseries (daily).csv"

        self.data_path = Path(data_path)
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.athlete_data: Dict[int, pd.DataFrame] = {}
        self.stats: Optional[RunnersDatasetStats] = None

    def load(self, verbose: bool = True) -> pd.DataFrame:
        """
        Load and process the dataset.

        Args:
            verbose: Print loading information

        Returns:
            Processed DataFrame with ACWR calculations
        """
        if verbose:
            print("=" * 70)
            print("LOADING MID-LONG DISTANCE RUNNERS DATASET")
            print("=" * 70)

        # Load raw data
        self.raw_data = pd.read_csv(self.data_path)

        if verbose:
            print(f"Loaded {len(self.raw_data):,} rows from {self.data_path.name}")

        # The dataset has 7 days of lagged data per row
        # Current day columns have no suffix, day-1 has ".1", day-2 has ".2", etc.
        # We need to extract the current day's metrics

        # Create processed dataset
        df = pd.DataFrame()

        # Extract key columns (current day = no suffix)
        df['athlete_id'] = self.raw_data['Athlete ID']
        df['date_index'] = self.raw_data['Date']  # Day index within athlete
        df['injury'] = self.raw_data['injury']

        # Training load metrics (current day)
        df['nr_sessions'] = self.raw_data['nr. sessions']
        df['total_km'] = self.raw_data['total km']
        df['km_z34'] = self.raw_data['km Z3-4']
        df['km_z5_t12'] = self.raw_data['km Z5-T1-T2']
        df['km_sprinting'] = self.raw_data['km sprinting']
        df['strength_training'] = self.raw_data['strength training']
        df['hours_alternative'] = self.raw_data['hours alternative']
        df['perceived_exertion'] = self.raw_data['perceived exertion']
        df['perceived_training_success'] = self.raw_data['perceived trainingSuccess']
        df['perceived_recovery'] = self.raw_data['perceived recovery']

        # Calculate training load (use total km as primary load metric)
        # Handle missing/invalid values
        df['load'] = df['total_km'].fillna(0).clip(lower=0)

        # Calculate weighted load (incorporating intensity zones)
        # Zone weighting: Z3-4 = 1.5x, Z5-T1-T2 = 2x, sprinting = 2.5x
        df['weighted_load'] = (
            df['load'] +
            0.5 * df['km_z34'].fillna(0).clip(lower=0) +  # Additional 0.5x for Z3-4
            1.0 * df['km_z5_t12'].fillna(0).clip(lower=0) +  # Additional 1x for Z5
            1.5 * df['km_sprinting'].fillna(0).clip(lower=0)  # Additional 1.5x for sprints
        )

        self.processed_data = df

        # Calculate ACWR for each athlete
        self._calculate_acwr(verbose)

        # Calculate statistics
        self._calculate_stats(verbose)

        return self.processed_data

    def _calculate_acwr(self, verbose: bool = True) -> None:
        """Calculate ACWR using rolling windows for each athlete."""
        if verbose:
            print("\nCalculating ACWR (7-day acute / 28-day chronic)...")

        acwr_values = []
        acute_values = []
        chronic_values = []

        for athlete_id in self.processed_data['athlete_id'].unique():
            mask = self.processed_data['athlete_id'] == athlete_id
            athlete_df = self.processed_data.loc[mask].copy()
            athlete_df = athlete_df.sort_values('date_index')

            # Calculate rolling means for load
            load = athlete_df['load'].values

            # Calculate EWMA (exponentially weighted moving average)
            # Using traditional rolling means for now
            n = len(load)
            athlete_acwr = []
            athlete_acute = []
            athlete_chronic = []

            for i in range(n):
                # Acute load (last 7 days)
                start_acute = max(0, i - 6)
                acute = np.mean(load[start_acute:i+1]) if i >= 0 else np.nan

                # Chronic load (last 28 days)
                start_chronic = max(0, i - 27)
                chronic = np.mean(load[start_chronic:i+1]) if i >= 6 else np.nan

                # ACWR
                if chronic is not None and chronic > 0.1:  # Avoid division by very small values
                    acwr = acute / chronic
                else:
                    acwr = np.nan

                athlete_acwr.append(acwr)
                athlete_acute.append(acute)
                athlete_chronic.append(chronic)

            acwr_values.extend(athlete_acwr)
            acute_values.extend(athlete_acute)
            chronic_values.extend(athlete_chronic)

            # Store athlete data
            athlete_df_copy = athlete_df.copy()
            athlete_df_copy['acwr'] = athlete_acwr
            athlete_df_copy['acute_load'] = athlete_acute
            athlete_df_copy['chronic_load'] = athlete_chronic
            self.athlete_data[athlete_id] = athlete_df_copy

        self.processed_data['acwr'] = acwr_values
        self.processed_data['acute_load'] = acute_values
        self.processed_data['chronic_load'] = chronic_values

        if verbose:
            valid_acwr = self.processed_data['acwr'].dropna()
            print(f"Valid ACWR values: {len(valid_acwr):,}")
            print(f"ACWR range: {valid_acwr.min():.3f} - {valid_acwr.max():.3f}")
            print(f"ACWR mean: {valid_acwr.mean():.3f}")

    def _calculate_stats(self, verbose: bool = True) -> None:
        """Calculate dataset statistics."""
        df = self.processed_data

        n_athletes = df['athlete_id'].nunique()
        n_days = len(df)
        n_injuries = df['injury'].sum()
        injury_rate = n_injuries / n_days * 100
        date_range = (df['date_index'].min(), df['date_index'].max())
        avg_days = n_days / n_athletes

        self.stats = RunnersDatasetStats(
            n_athletes=n_athletes,
            n_days=n_days,
            n_injuries=n_injuries,
            injury_rate=injury_rate,
            date_range=date_range,
            avg_days_per_athlete=avg_days
        )

        if verbose:
            print(f"\nDataset Statistics:")
            print(f"  Athletes: {n_athletes}")
            print(f"  Total days: {n_days:,}")
            print(f"  Injuries: {n_injuries}")
            print(f"  Injury rate: {injury_rate:.3f}%")
            print(f"  Avg days/athlete: {avg_days:.1f}")

    def get_acwr_injury_analysis(self, zone_boundaries: List[float] = None) -> pd.DataFrame:
        """
        Analyze injury rates by ACWR zone.

        Args:
            zone_boundaries: ACWR zone boundaries [low, optimal_high, caution, critical]
                           Default: [0.8, 1.3, 1.5, 2.0]

        Returns:
            DataFrame with injury analysis by zone
        """
        if zone_boundaries is None:
            zone_boundaries = [0.8, 1.3, 1.5, 2.0]

        df = self.processed_data.dropna(subset=['acwr']).copy()

        # Classify zones
        def classify_zone(acwr):
            if acwr >= zone_boundaries[3]:
                return 'critical'
            elif acwr >= zone_boundaries[2]:
                return 'high'
            elif acwr >= zone_boundaries[1]:
                return 'caution'
            elif acwr >= zone_boundaries[0]:
                return 'optimal'
            else:
                return 'low'

        df['zone'] = df['acwr'].apply(classify_zone)

        # Calculate injury rates by zone
        analysis = df.groupby('zone').agg(
            n_days=('injury', 'count'),
            n_injuries=('injury', 'sum'),
            mean_acwr=('acwr', 'mean')
        ).reset_index()

        analysis['injury_rate_pct'] = analysis['n_injuries'] / analysis['n_days'] * 100
        analysis['injury_rate_per_1000'] = analysis['n_injuries'] / analysis['n_days'] * 1000

        # Calculate relative risk (vs optimal zone)
        optimal_rate = analysis[analysis['zone'] == 'optimal']['injury_rate_pct'].values
        if len(optimal_rate) > 0 and optimal_rate[0] > 0:
            analysis['relative_risk'] = analysis['injury_rate_pct'] / optimal_rate[0]
        else:
            analysis['relative_risk'] = np.nan

        # Order zones
        zone_order = ['low', 'optimal', 'caution', 'high', 'critical']
        analysis['zone'] = pd.Categorical(analysis['zone'], categories=zone_order, ordered=True)
        analysis = analysis.sort_values('zone')

        return analysis

    def calculate_relative_risk_ci(self,
                                    threshold: float = 1.5,
                                    alpha: float = 0.05) -> Dict:
        """
        Calculate relative risk with confidence interval using Fisher's exact test.

        Args:
            threshold: ACWR threshold for high vs low risk
            alpha: Significance level for CI

        Returns:
            Dict with RR, CI, p-value, and sample sizes
        """
        from scipy import stats

        df = self.processed_data.dropna(subset=['acwr']).copy()

        # Split by threshold
        high_acwr = df[df['acwr'] >= threshold]
        low_acwr = df[df['acwr'] < threshold]

        # 2x2 contingency table
        a = high_acwr['injury'].sum()  # Injuries at high ACWR
        b = len(high_acwr) - a          # No injury at high ACWR
        c = low_acwr['injury'].sum()    # Injuries at low ACWR
        d = len(low_acwr) - c           # No injury at low ACWR

        # Injury rates
        rate_high = a / (a + b) if (a + b) > 0 else 0
        rate_low = c / (c + d) if (c + d) > 0 else 0

        # Relative risk
        if rate_low > 0:
            rr = rate_high / rate_low
        else:
            rr = np.inf

        # Confidence interval using log transformation
        if a > 0 and (a + b) > 0 and c > 0 and (c + d) > 0:
            se_log_rr = np.sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
            z = stats.norm.ppf(1 - alpha/2)
            log_rr = np.log(rr)
            ci_lower = np.exp(log_rr - z * se_log_rr)
            ci_upper = np.exp(log_rr + z * se_log_rr)
        else:
            ci_lower = np.nan
            ci_upper = np.nan

        # Fisher's exact test for p-value
        table = [[a, b], [c, d]]
        _, p_value = stats.fisher_exact(table, alternative='two-sided')

        return {
            'relative_risk': rr,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value,
            'threshold': threshold,
            'n_high_acwr': len(high_acwr),
            'n_low_acwr': len(low_acwr),
            'injuries_high_acwr': int(a),
            'injuries_low_acwr': int(c),
            'rate_high_acwr': rate_high,
            'rate_low_acwr': rate_low,
            'statistically_significant': p_value < alpha
        }

    def get_athlete_ids(self) -> List[int]:
        """Get list of athlete IDs."""
        return list(self.processed_data['athlete_id'].unique())

    def get_athlete_data(self, athlete_id: int) -> pd.DataFrame:
        """Get data for a specific athlete."""
        if athlete_id in self.athlete_data:
            return self.athlete_data[athlete_id]
        return self.processed_data[self.processed_data['athlete_id'] == athlete_id]

    def train_test_split_athletes(self,
                                   test_size: float = 0.3,
                                   random_state: int = 42) -> Tuple[List[int], List[int]]:
        """
        Split athletes into train/test sets.

        Args:
            test_size: Proportion of athletes for test set
            random_state: Random seed

        Returns:
            Tuple of (train_athlete_ids, test_athlete_ids)
        """
        np.random.seed(random_state)
        athlete_ids = self.get_athlete_ids()
        np.random.shuffle(athlete_ids)

        n_test = int(len(athlete_ids) * test_size)
        test_ids = athlete_ids[:n_test]
        train_ids = athlete_ids[n_test:]

        return train_ids, test_ids

    def get_train_test_data(self,
                            train_ids: List[int],
                            test_ids: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get train and test DataFrames based on athlete IDs.

        Args:
            train_ids: List of training athlete IDs
            test_ids: List of test athlete IDs

        Returns:
            Tuple of (train_df, test_df)
        """
        train_df = self.processed_data[self.processed_data['athlete_id'].isin(train_ids)]
        test_df = self.processed_data[self.processed_data['athlete_id'].isin(test_ids)]
        return train_df, test_df


def run_loader_validation():
    """Run validation on the loader."""
    print("=" * 70)
    print("RUNNERS DATA LOADER VALIDATION")
    print("=" * 70)

    loader = RunnersDataLoader()
    data = loader.load(verbose=True)

    print("\n" + "=" * 70)
    print("INJURY ANALYSIS BY ACWR ZONE")
    print("=" * 70)

    analysis = loader.get_acwr_injury_analysis()
    print(analysis.to_string(index=False))

    print("\n" + "=" * 70)
    print("RELATIVE RISK ANALYSIS (ACWR >= 1.5)")
    print("=" * 70)

    rr_analysis = loader.calculate_relative_risk_ci(threshold=1.5)
    print(f"Relative Risk: {rr_analysis['relative_risk']:.3f}")
    print(f"95% CI: [{rr_analysis['ci_lower']:.3f}, {rr_analysis['ci_upper']:.3f}]")
    print(f"p-value: {rr_analysis['p_value']:.6f}")
    print(f"Statistically significant: {rr_analysis['statistically_significant']}")
    print(f"\nHigh ACWR (>= 1.5):")
    print(f"  N = {rr_analysis['n_high_acwr']:,} days")
    print(f"  Injuries = {rr_analysis['injuries_high_acwr']}")
    print(f"  Rate = {rr_analysis['rate_high_acwr']*100:.3f}%")
    print(f"\nLow ACWR (< 1.5):")
    print(f"  N = {rr_analysis['n_low_acwr']:,} days")
    print(f"  Injuries = {rr_analysis['injuries_low_acwr']}")
    print(f"  Rate = {rr_analysis['rate_low_acwr']*100:.3f}%")

    print("\n" + "=" * 70)
    print("TRAIN/TEST SPLIT VALIDATION")
    print("=" * 70)

    train_ids, test_ids = loader.train_test_split_athletes(test_size=0.3)
    print(f"Train athletes: {len(train_ids)}")
    print(f"Test athletes: {len(test_ids)}")

    train_df, test_df = loader.get_train_test_data(train_ids, test_ids)
    print(f"Train days: {len(train_df):,}")
    print(f"Test days: {len(test_df):,}")

    return loader


if __name__ == "__main__":
    loader = run_loader_validation()
