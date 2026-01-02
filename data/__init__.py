"""Data generation and loading utilities."""

from .synthetic import RunnerProfile, generate_runner_profiles, generate_training_week
from .kaggle_loader import (
    KaggleDataLoader,
    KaggleDataLoaderPandas,
    KaggleDataLoaderPure,
    ActivitySummary,
    DailyTrimp,
    DOWNLOAD_INSTRUCTIONS,
)

__all__ = [
    # Synthetic data
    'RunnerProfile',
    'generate_runner_profiles',
    'generate_training_week',
    # Kaggle data loader
    'KaggleDataLoader',
    'KaggleDataLoaderPandas',
    'KaggleDataLoaderPure',
    'ActivitySummary',
    'DailyTrimp',
    'DOWNLOAD_INSTRUCTIONS',
]
