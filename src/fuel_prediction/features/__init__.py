"""Feature engineering module for aircraft fuel prediction."""

from .physics import (
    haversine_distance,
    get_atmosphere_properties,
    calculate_mach_number,
    add_physics_features
)
from .aggregations import aggregate_interval_features, clip_outliers

__all__ = [
    'haversine_distance',
    'get_atmosphere_properties',
    'calculate_mach_number',
    'add_physics_features',
    'aggregate_interval_features',
    'clip_outliers',
]
