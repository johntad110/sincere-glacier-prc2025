"""
Data loading utilities for fuel prediction.
"""
import os
import glob
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple
from joblib import Parallel, delayed

from ..utils.constants import RARE_AC_MAP, TRAJECTORY_COLUMNS_TO_INTERPOLATE
from ..features.physics import haversine_distance, add_physics_features
from ..features.aggregations import clip_outliers


def load_fuel_data(data_dir: str, dataset: str = 'train') -> pd.DataFrame:
    """
    Load fuel consumption data.
    
    Args:
        data_dir: Path to data directory
        dataset: Dataset name ('train', 'rank', 'final')
        
    Returns:
        Fuel data DataFrame
    """
    fuel_path = os.path.join(data_dir, f'fuel_{dataset}.parquet')
    return pd.read_parquet(fuel_path)


def load_flightlist(data_dir: str, dataset: str = 'train') -> pd.DataFrame:
    """
    Load flightlist metadata with aircraft type mapping applied.
    
    Args:
        data_dir: Path to data directory
        dataset: Dataset name ('train', 'rank', 'final')
        
    Returns:
        Flightlist DataFrame with mapped aircraft types
    """
    flightlist_path = os.path.join(data_dir, f'flightlist_{dataset}.parquet')
    df = pd.read_parquet(flightlist_path)
    
    # Apply rare aircraft mapping
    df['aircraft_type'] = df['aircraft_type'].replace(RARE_AC_MAP)
    
    return df


def load_airports(data_dir: str) -> pd.DataFrame:
    """
    Load airport reference data.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        Airport DataFrame
    """
    apt_path = os.path.join(data_dir, 'apt.parquet')
    return pd.read_parquet(apt_path)


def load_trajectory(traj_path: str) -> Optional[pd.DataFrame]:
    """
    Load and preprocess a single trajectory file.
    
    Args:
        traj_path: Path to trajectory parquet file
        
    Returns:
        Preprocessed trajectory DataFrame or None if loading fails
    """
    try:
        df = pd.read_parquet(traj_path)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Interpolate missing values
        df[TRAJECTORY_COLUMNS_TO_INTERPOLATE] = df[TRAJECTORY_COLUMNS_TO_INTERPOLATE].interpolate(
            method='linear', limit_direction='both'
        ).fillna(0)
        
        # Calculate step distance
        lats = df['latitude'].values
        lons = df['longitude'].values
        import numpy as np
        dist_step = np.zeros(len(df))
        dist_step[1:] = haversine_distance(
            lats[:-1], lons[:-1], lats[1:], lons[1:]
        )
        df['dist_step'] = dist_step
        
        # Add physics features
        df = add_physics_features(df)
        
        # Clip outliers
        df = clip_outliers(df)
        
        return df
        
    except Exception as e:
        print(f"Error loading {traj_path}: {e}")
        return None


def get_trajectory_paths(data_dir: str, dataset: str = 'train') -> List[str]:
    """
    Get list of trajectory file paths.
    
    Args:
        data_dir: Path to data directory
        dataset: Dataset name ('train', 'rank', 'final')
        
    Returns:
        List of trajectory file paths
    """
    flights_dir = os.path.join(data_dir, f'flights_{dataset}')
    return glob.glob(os.path.join(flights_dir, '*.parquet'))


def add_airport_features(
    flightlist: pd.DataFrame, 
    airports: pd.DataFrame
) -> pd.DataFrame:
    """
    Add origin and destination airport features to flightlist.
    
    Args:
        flightlist: Flightlist DataFrame
        airports: Airport reference DataFrame
        
    Returns:
        Flightlist with airport features added
    """
    # Merge origin coordinates
    flightlist = flightlist.merge(
        airports[['icao', 'latitude', 'longitude', 'elevation']].rename(columns={
            'latitude': 'origin_lat',
            'longitude': 'origin_lon',
            'elevation': 'origin_elev'
        }),
        left_on='origin_icao', right_on='icao', how='left'
    ).drop(columns=['icao'])
    
    # Merge destination coordinates
    flightlist = flightlist.merge(
        airports[['icao', 'latitude', 'longitude', 'elevation']].rename(columns={
            'latitude': 'dest_lat',
            'longitude': 'dest_lon',
            'elevation': 'dest_elev'
        }),
        left_on='destination_icao', right_on='icao', how='left'
    ).drop(columns=['icao'])
    
    # Calculate O-D distance
    flightlist['od_distance'] = haversine_distance(
        flightlist['origin_lat'], flightlist['origin_lon'],
        flightlist['dest_lat'], flightlist['dest_lon']
    )
    
    # Fill missing distances with mean
    flightlist['od_distance'] = flightlist['od_distance'].fillna(
        flightlist['od_distance'].mean()
    )
    
    return flightlist


def load_mass_proxy(data_dir: str) -> Optional[pd.DataFrame]:
    """
    Load mass proxy data if available.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        Mass proxy DataFrame or None if not found
    """
    mass_path = os.path.join(data_dir, 'mass_proxy.parquet')
    if os.path.exists(mass_path):
        return pd.read_parquet(mass_path)
    return None
