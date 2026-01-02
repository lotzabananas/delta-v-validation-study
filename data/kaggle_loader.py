"""
Kaggle Running and Heart Rate Data Loader.

Loads and transforms the mcandocia "Running and Heart Rate Data" dataset
from Kaggle for use with the Delta V backtesting framework.

Dataset: https://www.kaggle.com/datasets/mcandocia/running-heart-rate-recovery
License: CC BY-SA 4.0

The dataset contains three types of CSV files per activity:
1. Track data: Point-by-point GPS, HR, pace data
2. Lap data (_laps.csv): Lap-level summaries
3. Start/stop data (_starts.csv): Session event markers

For TRIMP calculation, we primarily use the lap data which provides:
- total_elapsed_time, total_timer_time (duration)
- avg_heart_rate, max_heart_rate
- total_distance, avg_speed
"""

import os
import glob
import zipfile
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ===============================================================================
# CONFIGURATION
# ===============================================================================

# Default athlete parameters (single-athlete dataset from Max Candocia)
# Based on dataset context: recreational male runner
DEFAULT_AGE = 30  # Estimated from dataset context
DEFAULT_GENDER = 'male'
DEFAULT_HR_REST = 60  # Typical resting HR
DEFAULT_HR_MAX = 190  # Will be overridden if data shows higher


# ===============================================================================
# DATA STRUCTURES
# ===============================================================================

@dataclass
class ActivitySummary:
    """Summary of a single running activity."""
    date: datetime
    duration_min: float
    avg_heart_rate: float
    max_heart_rate: float
    distance_m: float
    avg_speed_mps: float
    total_calories: float
    avg_temperature: Optional[float]
    source_file: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'date': self.date,
            'duration_min': self.duration_min,
            'avg_heart_rate': self.avg_heart_rate,
            'max_heart_rate': self.max_heart_rate,
            'distance_m': self.distance_m,
            'avg_speed_mps': self.avg_speed_mps,
            'total_calories': self.total_calories,
            'avg_temperature': self.avg_temperature,
            'source_file': self.source_file,
        }


@dataclass
class DailyTrimp:
    """Daily TRIMP value for backtesting."""
    date: datetime
    trimp: float
    duration_min: float
    avg_hr: float
    num_sessions: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'date': self.date,
            'trimp': self.trimp,
            'duration_min': self.duration_min,
            'avg_hr': self.avg_hr,
            'num_sessions': self.num_sessions,
        }


# ===============================================================================
# TRIMP CALCULATION
# ===============================================================================

def calculate_trimp(
    duration_min: float,
    hr_avg: float,
    hr_rest: float = DEFAULT_HR_REST,
    hr_max: float = DEFAULT_HR_MAX,
    gender: str = DEFAULT_GENDER
) -> float:
    """
    Calculate Training Impulse (TRIMP) for a session.

    TRIMP = Duration x Delta_HR x Y
    where:
        Delta_HR = (HR_avg - HR_rest) / (HR_max - HR_rest)
        Y = 0.64 * e^(1.92 * Delta_HR) for males
        Y = 0.86 * e^(1.67 * Delta_HR) for females

    Args:
        duration_min: Session duration in minutes
        hr_avg: Average heart rate during session
        hr_rest: Resting heart rate
        hr_max: Maximum heart rate
        gender: 'male' or 'female'

    Returns:
        TRIMP value
    """
    if hr_max <= hr_rest:
        return 0.0

    if hr_avg < hr_rest:
        hr_avg = hr_rest  # Can't be below resting

    # Calculate heart rate reserve fraction
    delta_hr = (hr_avg - hr_rest) / (hr_max - hr_rest)
    delta_hr = np.clip(delta_hr, 0.0, 1.0)

    # Calculate Y factor (gender-specific exponential weighting)
    if gender.lower() == 'male':
        y_factor = 0.64 * np.exp(1.92 * delta_hr)
    else:
        y_factor = 0.86 * np.exp(1.67 * delta_hr)

    # TRIMP = D x Delta_HR x Y
    trimp = duration_min * delta_hr * y_factor

    return trimp


# ===============================================================================
# DATA LOADING - PANDAS VERSION
# ===============================================================================

class KaggleDataLoaderPandas:
    """
    Load and transform Kaggle running data using pandas.

    Usage:
        loader = KaggleDataLoaderPandas('/path/to/kaggle/data')
        activities = loader.load_activities()
        daily_trimp = loader.calculate_daily_trimp()
    """

    def __init__(
        self,
        data_path: str,
        hr_rest: float = DEFAULT_HR_REST,
        hr_max: Optional[float] = None,
        age: int = DEFAULT_AGE,
        gender: str = DEFAULT_GENDER,
    ):
        """
        Initialize the loader.

        Args:
            data_path: Path to extracted Kaggle data directory or ZIP file
            hr_rest: Resting heart rate
            hr_max: Maximum heart rate (if None, estimated from age or data)
            age: Athlete age (for HR max estimation)
            gender: 'male' or 'female'
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for KaggleDataLoaderPandas")

        self.data_path = Path(data_path)
        self.hr_rest = hr_rest
        self.hr_max = hr_max if hr_max else (220 - age)
        self.age = age
        self.gender = gender

        self._activities: Optional[List[ActivitySummary]] = None
        self._daily_trimp: Optional[List[DailyTrimp]] = None

    def _extract_if_zip(self) -> Path:
        """Extract ZIP file if needed, return path to data directory."""
        if self.data_path.suffix == '.zip':
            extract_dir = self.data_path.parent / self.data_path.stem
            if not extract_dir.exists():
                with zipfile.ZipFile(self.data_path, 'r') as zf:
                    zf.extractall(extract_dir)
            return extract_dir
        return self.data_path

    def _find_lap_files(self) -> List[Path]:
        """Find all lap CSV files in the data directory."""
        data_dir = self._extract_if_zip()

        # Look for lap files (contain most useful summary data)
        lap_files = list(data_dir.rglob('*_laps.csv'))

        if not lap_files:
            # Fall back to track files
            lap_files = list(data_dir.rglob('*.csv'))
            lap_files = [f for f in lap_files
                        if '_starts' not in f.name and '_laps' not in f.name]

        return sorted(lap_files)

    def _parse_timestamp(self, ts: str) -> datetime:
        """Parse various timestamp formats."""
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%SZ',
        ]
        for fmt in formats:
            try:
                return datetime.strptime(str(ts), fmt)
            except ValueError:
                continue

        # Try pandas timestamp parsing as fallback
        return pd.to_datetime(ts).to_pydatetime()

    def load_activities(self) -> List[ActivitySummary]:
        """
        Load all activities from the dataset.

        Returns:
            List of ActivitySummary objects
        """
        if self._activities is not None:
            return self._activities

        lap_files = self._find_lap_files()

        if not lap_files:
            raise FileNotFoundError(
                f"No CSV files found in {self.data_path}. "
                "Please download the dataset from Kaggle and extract it."
            )

        activities = []
        max_hr_seen = 0

        for lap_file in lap_files:
            try:
                df = pd.read_csv(lap_file)

                # Get activity date from filename or first timestamp
                # Filename format: running_YY-MM-DD_HH-MM-SS_laps.csv
                filename = lap_file.name

                if 'timestamp' in df.columns:
                    activity_date = self._parse_timestamp(df['timestamp'].iloc[0])
                elif 'start_time' in df.columns:
                    activity_date = self._parse_timestamp(df['start_time'].iloc[0])
                else:
                    # Parse from filename
                    try:
                        date_str = filename.split('_')[1]
                        activity_date = datetime.strptime(date_str, '%y-%m-%d')
                    except (IndexError, ValueError):
                        continue

                # Aggregate lap data to get activity totals
                if 'total_timer_time' in df.columns:
                    total_duration = df['total_timer_time'].sum()
                elif 'total_elapsed_time' in df.columns:
                    total_duration = df['total_elapsed_time'].sum()
                else:
                    continue  # Skip if no duration

                duration_min = total_duration / 60.0

                # Heart rate - weighted average by duration
                if 'avg_heart_rate' in df.columns:
                    # Weight by lap duration
                    if 'total_timer_time' in df.columns:
                        weights = df['total_timer_time']
                    else:
                        weights = pd.Series([1] * len(df))

                    avg_hr = np.average(
                        df['avg_heart_rate'].fillna(0),
                        weights=weights
                    )
                    max_hr = df['max_heart_rate'].max() if 'max_heart_rate' in df.columns else avg_hr
                else:
                    continue  # Skip if no HR data

                # Update max HR seen (for auto-calibration)
                if max_hr > max_hr_seen:
                    max_hr_seen = max_hr

                # Distance
                total_distance = df['total_distance'].sum() if 'total_distance' in df.columns else 0

                # Speed - weighted average
                if 'avg_speed' in df.columns or 'enhanced_avg_speed' in df.columns:
                    speed_col = 'enhanced_avg_speed' if 'enhanced_avg_speed' in df.columns else 'avg_speed'
                    avg_speed = np.average(
                        df[speed_col].fillna(0),
                        weights=weights
                    )
                else:
                    avg_speed = total_distance / total_duration if total_duration > 0 else 0

                # Calories and temperature
                total_calories = df['total_calories'].sum() if 'total_calories' in df.columns else 0
                avg_temp = df['avg_temperature'].mean() if 'avg_temperature' in df.columns else None

                activity = ActivitySummary(
                    date=activity_date,
                    duration_min=duration_min,
                    avg_heart_rate=avg_hr,
                    max_heart_rate=max_hr,
                    distance_m=total_distance,
                    avg_speed_mps=avg_speed,
                    total_calories=total_calories,
                    avg_temperature=avg_temp,
                    source_file=str(lap_file),
                )

                activities.append(activity)

            except Exception as e:
                print(f"Warning: Could not parse {lap_file}: {e}")
                continue

        # Auto-calibrate HR max if we saw higher values
        if max_hr_seen > self.hr_max:
            print(f"Note: Adjusting HR max from {self.hr_max} to {max_hr_seen + 5} based on data")
            self.hr_max = max_hr_seen + 5  # Add 5 bpm buffer

        # Sort by date
        activities.sort(key=lambda a: a.date)

        self._activities = activities
        return activities

    def calculate_daily_trimp(self) -> List[DailyTrimp]:
        """
        Calculate daily TRIMP values from activities.

        Multiple activities on the same day are summed.
        Missing days (rest days) get TRIMP = 0.

        Returns:
            List of DailyTrimp objects for each day in the date range
        """
        if self._daily_trimp is not None:
            return self._daily_trimp

        activities = self.load_activities()

        if not activities:
            return []

        # Group activities by date (ignoring time)
        daily_data: Dict[datetime, List[ActivitySummary]] = {}
        for activity in activities:
            date_key = activity.date.replace(hour=0, minute=0, second=0, microsecond=0)
            if date_key not in daily_data:
                daily_data[date_key] = []
            daily_data[date_key].append(activity)

        # Get date range
        min_date = min(daily_data.keys())
        max_date = max(daily_data.keys())

        # Generate daily TRIMP values
        daily_trimp = []
        current_date = min_date

        while current_date <= max_date:
            if current_date in daily_data:
                day_activities = daily_data[current_date]

                # Sum TRIMP for all activities
                total_trimp = 0.0
                total_duration = 0.0
                weighted_hr_sum = 0.0

                for activity in day_activities:
                    trimp = calculate_trimp(
                        duration_min=activity.duration_min,
                        hr_avg=activity.avg_heart_rate,
                        hr_rest=self.hr_rest,
                        hr_max=self.hr_max,
                        gender=self.gender,
                    )
                    total_trimp += trimp
                    total_duration += activity.duration_min
                    weighted_hr_sum += activity.avg_heart_rate * activity.duration_min

                avg_hr = weighted_hr_sum / total_duration if total_duration > 0 else 0

                daily_trimp.append(DailyTrimp(
                    date=current_date,
                    trimp=total_trimp,
                    duration_min=total_duration,
                    avg_hr=avg_hr,
                    num_sessions=len(day_activities),
                ))
            else:
                # Rest day
                daily_trimp.append(DailyTrimp(
                    date=current_date,
                    trimp=0.0,
                    duration_min=0.0,
                    avg_hr=0.0,
                    num_sessions=0,
                ))

            current_date += timedelta(days=1)

        self._daily_trimp = daily_trimp
        return daily_trimp

    def get_trimp_array(self) -> np.ndarray:
        """
        Get daily TRIMP values as a numpy array.

        This is the primary output format for the backtesting framework.

        Returns:
            Array of daily TRIMP values
        """
        daily_trimp = self.calculate_daily_trimp()
        return np.array([d.trimp for d in daily_trimp])

    def to_dataframe(self) -> 'pd.DataFrame':
        """
        Convert daily TRIMP data to a pandas DataFrame.

        Returns:
            DataFrame with columns: date, trimp, duration_min, avg_hr, num_sessions
        """
        daily_trimp = self.calculate_daily_trimp()
        return pd.DataFrame([d.to_dict() for d in daily_trimp])

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics about the loaded data.

        Returns:
            Dictionary with summary statistics
        """
        activities = self.load_activities()
        daily_trimp = self.calculate_daily_trimp()

        if not activities:
            return {'error': 'No activities loaded'}

        trimp_values = [d.trimp for d in daily_trimp if d.trimp > 0]

        return {
            'num_activities': len(activities),
            'date_range': {
                'start': min(a.date for a in activities).isoformat(),
                'end': max(a.date for a in activities).isoformat(),
            },
            'total_days': len(daily_trimp),
            'active_days': sum(1 for d in daily_trimp if d.trimp > 0),
            'rest_days': sum(1 for d in daily_trimp if d.trimp == 0),
            'trimp_stats': {
                'mean': np.mean(trimp_values) if trimp_values else 0,
                'std': np.std(trimp_values) if trimp_values else 0,
                'min': np.min(trimp_values) if trimp_values else 0,
                'max': np.max(trimp_values) if trimp_values else 0,
                'total': sum(trimp_values),
            },
            'duration_stats': {
                'mean_min': np.mean([a.duration_min for a in activities]),
                'total_hours': sum(a.duration_min for a in activities) / 60,
            },
            'hr_stats': {
                'mean_avg': np.mean([a.avg_heart_rate for a in activities]),
                'max_seen': max(a.max_heart_rate for a in activities),
            },
            'parameters': {
                'hr_rest': self.hr_rest,
                'hr_max': self.hr_max,
                'gender': self.gender,
            },
        }


# ===============================================================================
# DATA LOADING - PURE PYTHON VERSION (NO PANDAS)
# ===============================================================================

class KaggleDataLoaderPure:
    """
    Load and transform Kaggle running data without pandas dependency.

    Provides the same interface as KaggleDataLoaderPandas but uses
    only standard library modules.
    """

    def __init__(
        self,
        data_path: str,
        hr_rest: float = DEFAULT_HR_REST,
        hr_max: Optional[float] = None,
        age: int = DEFAULT_AGE,
        gender: str = DEFAULT_GENDER,
    ):
        """Initialize the loader."""
        self.data_path = Path(data_path)
        self.hr_rest = hr_rest
        self.hr_max = hr_max if hr_max else (220 - age)
        self.age = age
        self.gender = gender

        self._activities: Optional[List[ActivitySummary]] = None
        self._daily_trimp: Optional[List[DailyTrimp]] = None

    def _extract_if_zip(self) -> Path:
        """Extract ZIP file if needed."""
        if self.data_path.suffix == '.zip':
            extract_dir = self.data_path.parent / self.data_path.stem
            if not extract_dir.exists():
                with zipfile.ZipFile(self.data_path, 'r') as zf:
                    zf.extractall(extract_dir)
            return extract_dir
        return self.data_path

    def _find_lap_files(self) -> List[Path]:
        """Find all lap CSV files."""
        import csv
        data_dir = self._extract_if_zip()

        lap_files = []
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if f.endswith('_laps.csv'):
                    lap_files.append(Path(root) / f)

        return sorted(lap_files)

    def _parse_csv(self, filepath: Path) -> List[Dict[str, str]]:
        """Parse a CSV file into a list of dictionaries."""
        import csv
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)

    def _safe_float(self, value: str, default: float = 0.0) -> float:
        """Safely convert string to float."""
        try:
            return float(value) if value else default
        except (ValueError, TypeError):
            return default

    def _parse_timestamp(self, ts: str) -> datetime:
        """Parse timestamp string."""
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M:%S.%f',
        ]
        for fmt in formats:
            try:
                return datetime.strptime(str(ts), fmt)
            except ValueError:
                continue
        raise ValueError(f"Cannot parse timestamp: {ts}")

    def load_activities(self) -> List[ActivitySummary]:
        """Load all activities from the dataset."""
        if self._activities is not None:
            return self._activities

        lap_files = self._find_lap_files()

        if not lap_files:
            raise FileNotFoundError(
                f"No lap CSV files found in {self.data_path}"
            )

        activities = []
        max_hr_seen = 0

        for lap_file in lap_files:
            try:
                rows = self._parse_csv(lap_file)
                if not rows:
                    continue

                # Get activity date
                first_row = rows[0]
                if 'timestamp' in first_row:
                    activity_date = self._parse_timestamp(first_row['timestamp'])
                elif 'start_time' in first_row:
                    activity_date = self._parse_timestamp(first_row['start_time'])
                else:
                    continue

                # Calculate totals
                total_duration = sum(
                    self._safe_float(r.get('total_timer_time', r.get('total_elapsed_time', 0)))
                    for r in rows
                )

                if total_duration == 0:
                    continue

                duration_min = total_duration / 60.0

                # Heart rate - weighted average
                hr_sum = 0.0
                max_hr = 0.0
                for r in rows:
                    lap_dur = self._safe_float(r.get('total_timer_time', 1))
                    lap_hr = self._safe_float(r.get('avg_heart_rate', 0))
                    lap_max_hr = self._safe_float(r.get('max_heart_rate', 0))
                    hr_sum += lap_hr * lap_dur
                    max_hr = max(max_hr, lap_max_hr)

                avg_hr = hr_sum / total_duration if total_duration > 0 else 0

                if avg_hr == 0:
                    continue  # Skip if no HR data

                if max_hr > max_hr_seen:
                    max_hr_seen = max_hr

                # Distance and speed
                total_distance = sum(
                    self._safe_float(r.get('total_distance', 0))
                    for r in rows
                )
                avg_speed = total_distance / total_duration if total_duration > 0 else 0

                # Other fields
                total_calories = sum(
                    self._safe_float(r.get('total_calories', 0))
                    for r in rows
                )

                temps = [self._safe_float(r.get('avg_temperature', 0)) for r in rows]
                avg_temp = sum(temps) / len(temps) if temps else None

                activity = ActivitySummary(
                    date=activity_date,
                    duration_min=duration_min,
                    avg_heart_rate=avg_hr,
                    max_heart_rate=max_hr,
                    distance_m=total_distance,
                    avg_speed_mps=avg_speed,
                    total_calories=total_calories,
                    avg_temperature=avg_temp,
                    source_file=str(lap_file),
                )

                activities.append(activity)

            except Exception as e:
                print(f"Warning: Could not parse {lap_file}: {e}")
                continue

        # Auto-calibrate HR max
        if max_hr_seen > self.hr_max:
            self.hr_max = max_hr_seen + 5

        activities.sort(key=lambda a: a.date)
        self._activities = activities
        return activities

    def calculate_daily_trimp(self) -> List[DailyTrimp]:
        """Calculate daily TRIMP values."""
        if self._daily_trimp is not None:
            return self._daily_trimp

        activities = self.load_activities()

        if not activities:
            return []

        # Group by date
        daily_data: Dict[datetime, List[ActivitySummary]] = {}
        for activity in activities:
            date_key = activity.date.replace(hour=0, minute=0, second=0, microsecond=0)
            if date_key not in daily_data:
                daily_data[date_key] = []
            daily_data[date_key].append(activity)

        # Date range
        min_date = min(daily_data.keys())
        max_date = max(daily_data.keys())

        # Generate daily values
        daily_trimp = []
        current_date = min_date

        while current_date <= max_date:
            if current_date in daily_data:
                day_activities = daily_data[current_date]

                total_trimp = 0.0
                total_duration = 0.0
                weighted_hr_sum = 0.0

                for activity in day_activities:
                    trimp = calculate_trimp(
                        duration_min=activity.duration_min,
                        hr_avg=activity.avg_heart_rate,
                        hr_rest=self.hr_rest,
                        hr_max=self.hr_max,
                        gender=self.gender,
                    )
                    total_trimp += trimp
                    total_duration += activity.duration_min
                    weighted_hr_sum += activity.avg_heart_rate * activity.duration_min

                avg_hr = weighted_hr_sum / total_duration if total_duration > 0 else 0

                daily_trimp.append(DailyTrimp(
                    date=current_date,
                    trimp=total_trimp,
                    duration_min=total_duration,
                    avg_hr=avg_hr,
                    num_sessions=len(day_activities),
                ))
            else:
                daily_trimp.append(DailyTrimp(
                    date=current_date,
                    trimp=0.0,
                    duration_min=0.0,
                    avg_hr=0.0,
                    num_sessions=0,
                ))

            current_date += timedelta(days=1)

        self._daily_trimp = daily_trimp
        return daily_trimp

    def get_trimp_array(self) -> np.ndarray:
        """Get daily TRIMP values as numpy array."""
        daily_trimp = self.calculate_daily_trimp()
        return np.array([d.trimp for d in daily_trimp])

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        activities = self.load_activities()
        daily_trimp = self.calculate_daily_trimp()

        if not activities:
            return {'error': 'No activities loaded'}

        trimp_values = [d.trimp for d in daily_trimp if d.trimp > 0]

        return {
            'num_activities': len(activities),
            'date_range': {
                'start': min(a.date for a in activities).isoformat(),
                'end': max(a.date for a in activities).isoformat(),
            },
            'total_days': len(daily_trimp),
            'active_days': sum(1 for d in daily_trimp if d.trimp > 0),
            'rest_days': sum(1 for d in daily_trimp if d.trimp == 0),
            'trimp_stats': {
                'mean': np.mean(trimp_values) if trimp_values else 0,
                'std': np.std(trimp_values) if trimp_values else 0,
                'min': np.min(trimp_values) if trimp_values else 0,
                'max': np.max(trimp_values) if trimp_values else 0,
                'total': sum(trimp_values),
            },
            'parameters': {
                'hr_rest': self.hr_rest,
                'hr_max': self.hr_max,
                'gender': self.gender,
            },
        }


# ===============================================================================
# CONVENIENCE FACTORY
# ===============================================================================

def KaggleDataLoader(
    data_path: str,
    hr_rest: float = DEFAULT_HR_REST,
    hr_max: Optional[float] = None,
    age: int = DEFAULT_AGE,
    gender: str = DEFAULT_GENDER,
    use_pandas: bool = True,
):
    """
    Factory function to create appropriate loader.

    Args:
        data_path: Path to Kaggle data
        hr_rest: Resting heart rate
        hr_max: Maximum heart rate
        age: Athlete age
        gender: 'male' or 'female'
        use_pandas: Whether to use pandas (faster, more features)

    Returns:
        KaggleDataLoaderPandas or KaggleDataLoaderPure instance
    """
    if use_pandas and HAS_PANDAS:
        return KaggleDataLoaderPandas(
            data_path=data_path,
            hr_rest=hr_rest,
            hr_max=hr_max,
            age=age,
            gender=gender,
        )
    else:
        return KaggleDataLoaderPure(
            data_path=data_path,
            hr_rest=hr_rest,
            hr_max=hr_max,
            age=age,
            gender=gender,
        )


# ===============================================================================
# DOWNLOAD INSTRUCTIONS
# ===============================================================================

DOWNLOAD_INSTRUCTIONS = """
================================================================================
KAGGLE DATASET DOWNLOAD INSTRUCTIONS
================================================================================

Dataset: "Running and Heart Rate Data" by mcandocia
URL: https://www.kaggle.com/datasets/mcandocia/running-heart-rate-recovery

METHOD 1: Web Download (Recommended)
-------------------------------------
1. Go to: https://www.kaggle.com/datasets/mcandocia/running-heart-rate-recovery
2. Click the "Download" button (you'll need a free Kaggle account)
3. Extract the ZIP file to: /Users/timmac/Desktop/Delta V backtesting/data/kaggle/

METHOD 2: Kaggle CLI
--------------------
1. Install kaggle CLI: pip install kaggle
2. Set up API credentials: https://www.kaggle.com/docs/api
3. Run: kaggle datasets download -d mcandocia/running-heart-rate-recovery
4. Extract to: /Users/timmac/Desktop/Delta V backtesting/data/kaggle/

After downloading, use the loader:

    from data.kaggle_loader import KaggleDataLoader

    loader = KaggleDataLoader(
        data_path='/Users/timmac/Desktop/Delta V backtesting/data/kaggle',
        hr_rest=60,  # Adjust to your resting HR
        age=30,      # Adjust to athlete age
        gender='male'
    )

    # Get daily TRIMP array for backtesting
    trimp_history = loader.get_trimp_array()

    # Or get full DataFrame
    df = loader.to_dataframe()

    # Summary stats
    print(loader.get_summary_stats())

================================================================================
"""


# ===============================================================================
# COMMAND LINE INTERFACE
# ===============================================================================

if __name__ == '__main__':
    import sys

    print(DOWNLOAD_INSTRUCTIONS)

    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        print(f"\nAttempting to load data from: {data_path}")

        try:
            loader = KaggleDataLoader(data_path)
            activities = loader.load_activities()

            print(f"\nLoaded {len(activities)} activities")
            print("\nFirst 5 activities:")
            for a in activities[:5]:
                print(f"  {a.date.strftime('%Y-%m-%d')}: {a.duration_min:.1f} min, "
                      f"HR {a.avg_heart_rate:.0f}/{a.max_heart_rate:.0f}, "
                      f"TRIMP {calculate_trimp(a.duration_min, a.avg_heart_rate):.1f}")

            print("\nSummary Statistics:")
            stats = loader.get_summary_stats()
            for key, value in stats.items():
                print(f"  {key}: {value}")

            # Show sample transformed data
            daily_trimp = loader.calculate_daily_trimp()
            print(f"\nDaily TRIMP array shape: ({len(daily_trimp)},)")
            print("First 14 days:")
            for d in daily_trimp[:14]:
                status = f"Run: {d.duration_min:.0f}min, HR {d.avg_hr:.0f}" if d.trimp > 0 else "Rest"
                print(f"  {d.date.strftime('%Y-%m-%d')}: TRIMP={d.trimp:6.1f} [{status}]")

        except Exception as e:
            print(f"Error: {e}")
            print("\nPlease download the dataset first. See instructions above.")
    else:
        print("\nUsage: python kaggle_loader.py <path_to_kaggle_data>")
        print("Example: python kaggle_loader.py ./data/kaggle/")
