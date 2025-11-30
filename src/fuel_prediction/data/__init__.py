"""Data loading and preprocessing utilities."""

from .loader import (
    load_fuel_data,
    load_flightlist,
    load_airports,
    load_trajectory,
    get_trajectory_paths,
    add_airport_features,
    load_mass_proxy
)

__all__ = [
    'load_fuel_data',
    'load_flightlist',
    'load_airports',
    'load_trajectory',
    'get_trajectory_paths',
    'add_airport_features',
    'load_mass_proxy',
]
