"""
Fuel Prediction Package for PRC 2025 Data Challenge.

A python package for predicting aircraft fuel consumption using
hybrid stacking of physics-aware GBM and sequence-aware LSTM models.
"""

__version__ = "1.0.0"
__author__ = "Team Sincere Glacier"

from .config import Config, load_config
from .utils import setup_logger, calculate_rmse

__all__ = [
    'Config',
    'load_config',
    'setup_logger',
    'calculate_rmse',
]
