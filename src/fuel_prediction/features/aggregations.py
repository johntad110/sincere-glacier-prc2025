"""
Statistical aggregation functions for feature engineering.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any

from ..utils.constants import (
    ACCELERATION_CLIP_RANGE,
    ENERGY_RATE_CLIP_RANGE
)


def aggregate_interval_features(
    df: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    idx: int,
    flight_id: str,
    fuel_kg: float,
    ac_type: str
) -> Dict[str, Any]:
    """
    Aggregate features for a single flight interval.
    
    Args:
        df: Trajectory dataframe with physics features
        start_time: Interval start timestamp
        end_time: Interval end timestamp
        idx: Interval index
        flight_id: Flight identifier
        fuel_kg: Target fuel consumption
        ac_type: Aircraft type
        
    Returns:
        Dictionary of aggregated features
    """
    mid_ts = start_time + (end_time - start_time) / 2
    mask = (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)
    
    # Base features
    features = {
        'idx': idx,
        'flight_id': flight_id,
        'timestamp_mid': mid_ts,
        'duration': (end_time - start_time).total_seconds(),
        'n_points': np.sum(mask),
        'fuel_kg': fuel_kg,
        'aircraft_type': ac_type
    }
    
    if not np.any(mask):
        # No data in interval - fill with NaN
        nan_features = {
            'avg_alt': np.nan, 'std_alt': np.nan, 'min_alt': np.nan, 'max_alt': np.nan,
            'avg_speed': np.nan, 'std_speed': np.nan,
            'total_dist': np.nan, 'avg_vertical_rate': np.nan,
            'avg_acc': np.nan, 'std_acc': np.nan, 'min_acc': np.nan, 'max_acc': np.nan,
            'avg_energy_rate': np.nan, 'std_energy_rate': np.nan, 
            'min_energy_rate': np.nan, 'max_energy_rate': np.nan,
            'avg_mach': np.nan, 'avg_dynamic_pressure': np.nan, 'avg_air_density': np.nan,
            'avg_parasitic_power': np.nan, 'avg_induced_power': np.nan, 'avg_climb_power': np.nan,
        }
        features.update(nan_features)
        return features
    
    # Slice data for interval
    interval_data = df[mask]
    
    # Altitude statistics
    features['avg_alt'] = interval_data['altitude'].mean()
    features['std_alt'] = interval_data['altitude'].std()
    features['min_alt'] = interval_data['altitude'].min()
    features['max_alt'] = interval_data['altitude'].max()
    
    # Speed statistics
    features['avg_speed'] = interval_data['groundspeed'].mean()
    features['std_speed'] = interval_data['groundspeed'].std()
    
    # Distance
    features['total_dist'] = interval_data['dist_step'].sum()
    
    # Vertical rate
    features['avg_vertical_rate'] = interval_data['vertical_rate'].mean()
    
    # Acceleration statistics
    features['avg_acc'] = interval_data['acceleration'].mean()
    features['std_acc'] = interval_data['acceleration'].std()
    features['min_acc'] = interval_data['acceleration'].min()
    features['max_acc'] = interval_data['acceleration'].max()
    
    # Energy rate statistics
    features['avg_energy_rate'] = interval_data['energy_rate'].mean()
    features['std_energy_rate'] = interval_data['energy_rate'].std()
    features['min_energy_rate'] = interval_data['energy_rate'].min()
    features['max_energy_rate'] = interval_data['energy_rate'].max()
    
    # Aerodynamic features
    features['avg_mach'] = interval_data['mach'].mean()
    features['avg_dynamic_pressure'] = interval_data['dynamic_pressure'].mean()
    features['avg_air_density'] = interval_data['air_density'].mean()
    features['avg_parasitic_power'] = interval_data['parasitic_power'].mean()
    features['avg_induced_power'] = interval_data['induced_power'].mean()
    features['avg_climb_power'] = interval_data['climb_power'].mean()
    
    return features


def clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clip outlier values in acceleration and energy rate.
    
    Args:
        df: DataFrame with acceleration and energy_rate columns
        
    Returns:
        DataFrame with clipped values
    """
    df['acceleration'] = df['acceleration'].clip(*ACCELERATION_CLIP_RANGE)
    df['energy_rate'] = df['energy_rate'].clip(*ENERGY_RATE_CLIP_RANGE)
    
    return df
