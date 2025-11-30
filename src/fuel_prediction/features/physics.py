"""
Physics-based feature engineering for aircraft fuel prediction.

This module contains functions for calculating physics-based features
from flight trajectory data, including aerodynamic and atmospheric properties.
"""
import numpy as np
import pandas as pd
from typing import Tuple

from ..utils.constants import (
    GRAVITY, GAS_CONSTANT, SEA_LEVEL_TEMP, SEA_LEVEL_PRESSURE,
    LAPSE_RATE, GAMMA, EARTH_RADIUS, KT_TO_MS, FT_TO_M
)


def haversine_distance(
    lat1: np.ndarray, 
    lon1: np.ndarray, 
    lat2: np.ndarray, 
    lon2: np.ndarray
) -> np.ndarray:
    """
    Calculate great circle distance between two points on Earth.
    
    Uses the Haversine formula to calculate the distance between two points
    on the Earth's surface given their latitude and longitude.
    
    Args:
        lat1: Latitude of first point (degrees)
        lon1: Longitude of first point (degrees)
        lat2: Latitude of second point (degrees)
        lon2: Longitude of second point (degrees)
        
    Returns:
        Distance in kilometers
    """
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return EARTH_RADIUS * c


def get_atmosphere_properties(altitude_ft: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate ISA standard atmosphere properties at given altitudes.
    
    Calculates air density and speed of sound using the International Standard
    Atmosphere (ISA) model. Simplified for troposphere with stratosphere cap.
    
    Args:
        altitude_ft: Altitude in feet
        
    Returns:
        Tuple of (air_density in kg/m³, speed_of_sound in m/s)
    """
    # Convert altitude to meters
    h_m = altitude_ft * FT_TO_M
    
    # Temperature (capped at stratosphere temperature)
    T = np.maximum(SEA_LEVEL_TEMP - LAPSE_RATE * h_m, 216.65)
    
    # Pressure (simplified for troposphere)
    P = SEA_LEVEL_PRESSURE * (1 - LAPSE_RATE * h_m / SEA_LEVEL_TEMP) ** (GRAVITY / (GAS_CONSTANT * LAPSE_RATE))
    
    # Density: ρ = P / (R * T)
    rho = P / (GAS_CONSTANT * T)
    
    # Speed of sound: a = sqrt(γ * R * T)
    a = np.sqrt(GAMMA * GAS_CONSTANT * T)
    
    return rho, a


def calculate_mach_number(
    groundspeed_kt: np.ndarray, 
    altitude_ft: np.ndarray
) -> np.ndarray:
    """
    Calculate Mach number from groundspeed and altitude.
    
    Args:
        groundspeed_kt: Groundspeed in knots
        altitude_ft: Altitude in feet
        
    Returns:
        Mach number (dimensionless)
    """
    v_ms = groundspeed_kt * KT_TO_MS
    _, a_sound = get_atmosphere_properties(altitude_ft)
    
    return v_ms / a_sound


def calculate_dynamic_pressure(
    groundspeed_kt: np.ndarray,
    altitude_ft: np.ndarray
) -> np.ndarray:
    """
    Calculate dynamic pressure (q = 0.5 * ρ * v²).
    
    Args:
        groundspeed_kt: Groundspeed in knots
        altitude_ft: Altitude in feet
        
    Returns:
        Dynamic pressure in Pa
    """
    v_ms = groundspeed_kt * KT_TO_MS
    rho, _ = get_atmosphere_properties(altitude_ft)
    
    return 0.5 * rho * (v_ms ** 2)


def calculate_acceleration(
    groundspeed_kt: np.ndarray,
    dt: np.ndarray
) -> np.ndarray:
    """
    Calculate acceleration from groundspeed time series.
    
    Args:
        groundspeed_kt: Groundspeed in knots
        dt: Time step in seconds
        
    Returns:
        Acceleration in m/s²
    """
    v_ms = groundspeed_kt * KT_TO_MS
    
    # Avoid division by zero
    dt_safe = np.copy(dt)
    dt_safe[dt_safe == 0] = 1.0
    
    acc = np.zeros(len(v_ms))
    acc[1:] = (v_ms[1:] - v_ms[:-1]) / dt_safe[1:]
    
    return acc


def calculate_energy_rate(
    vertical_rate_fpm: np.ndarray,
    acceleration_ms2: np.ndarray,
    groundspeed_kt: np.ndarray
) -> np.ndarray:
    """
    Calculate specific energy rate (rate of change of total energy per unit weight).
    
    E_rate = h_dot + (v/g) * a
    where h_dot is vertical rate and a is acceleration
    
    Args:
        vertical_rate_fpm: Vertical rate in feet per minute
        acceleration_ms2: Acceleration in m/s²
        groundspeed_kt: Groundspeed in knots
        
    Returns:
        Energy rate in m/s
    """
    h_dot_ms = vertical_rate_fpm * (FT_TO_M / 60.0)
    v_ms = groundspeed_kt * KT_TO_MS
    
    return h_dot_ms + (v_ms / GRAVITY) * acceleration_ms2


def calculate_parasitic_power(
    groundspeed_kt: np.ndarray,
    altitude_ft: np.ndarray
) -> np.ndarray:
    """
    Calculate parasitic power proxy (ρ * v³).
    
    Parasitic power is proportional to air density and velocity cubed.
    
    Args:
        groundspeed_kt: Groundspeed in knots
        altitude_ft: Altitude in feet
        
    Returns:
        Parasitic power proxy (arbitrary units)
    """
    v_ms = groundspeed_kt * KT_TO_MS
    rho, _ = get_atmosphere_properties(altitude_ft)
    
    return rho * (v_ms ** 3)


def calculate_induced_power(groundspeed_kt: np.ndarray) -> np.ndarray:
    """
    Calculate induced power proxy (1/v).
    
    Induced power is inversely proportional to velocity (simplified).
    
    Args:
        groundspeed_kt: Groundspeed in knots
        
    Returns:
        Induced power proxy (arbitrary units)
    """
    v_ms = groundspeed_kt * KT_TO_MS
    v_safe = np.maximum(v_ms, 1.0)  # Avoid division by zero
    
    return 1.0 / v_safe


def calculate_climb_power(vertical_rate_fpm: np.ndarray) -> np.ndarray:
    """
    Calculate climb power proxy (vertical speed).
    
    Args:
        vertical_rate_fpm: Vertical rate in feet per minute
        
    Returns:
        Climb power proxy in m/s
    """
    return vertical_rate_fpm * (FT_TO_M / 60.0)


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all physics-based features to trajectory dataframe.
    
    Calculates and adds columns for:
    - Mach number
    - Dynamic pressure
    - Air density
    - Acceleration
    - Energy rate
    - Parasitic power
    - Induced power
    - Climb power
    
    Args:
        df: Trajectory dataframe with columns: groundspeed, altitude, 
            vertical_rate, timestamp
            
    Returns:
        DataFrame with added physics features
    """
    # Calculate time delta
    df['dt'] = df['timestamp'].diff().dt.total_seconds().fillna(0)
    
    # Atmosphere properties
    rho, a_sound = get_atmosphere_properties(df['altitude'].values)
    df['air_density'] = rho
    df['speed_of_sound'] = a_sound
    
    # Velocity features
    v_ms = df['groundspeed'].values * KT_TO_MS
    df['mach'] = v_ms / a_sound
    df['dynamic_pressure'] = 0.5 * rho * (v_ms ** 2)
    
    # Acceleration
    df['acceleration'] = calculate_acceleration(
        df['groundspeed'].values, 
        df['dt'].values
    )
    
    # Energy rate
    df['energy_rate'] = calculate_energy_rate(
        df['vertical_rate'].values,
        df['acceleration'].values,
        df['groundspeed'].values
    )
    
    # Power proxies
    df['parasitic_power'] = calculate_parasitic_power(
        df['groundspeed'].values,
        df['altitude'].values
    )
    df['induced_power'] = calculate_induced_power(df['groundspeed'].values)
    df['climb_power'] = calculate_climb_power(df['vertical_rate'].values)
    
    return df
