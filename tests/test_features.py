"""
Test suite for physics-based features.
"""
import pytest
import numpy as np
import pandas as pd

from fuel_prediction.features.physics import (
    haversine_distance,
    get_atmosphere_properties,
    calculate_mach_number
)


class TestPhysics:
    """Test physics calculations."""
    
    def test_haversine_same_point(self):
        """Test distance between same point is zero."""
        lat, lon = 40.0, -74.0
        dist = haversine_distance(lat, lon, lat, lon)
        assert dist == 0.0
    
    def test_haversine_known_distance(self):
        """Test known distance (NYC to LA approximately)."""
        # NYC: 40.7128° N, 74.0060° W
        # LA: 34.0522° N, 118.2437° W
        dist = haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)
        
        # Approximate great circle distance: ~3,944 km
        assert 3900 < dist < 4000
    
    def test_atmosphere_sea_level(self):
        """Test ISA atmosphere at sea level."""
        rho, a = get_atmosphere_properties(0)  # Sea level
        
        # Expected: rho ≈ 1.225 kg/m³, a ≈ 340 m/s
        assert 1.2 < rho < 1.3
        assert 335 < a < 345
    
    def test_atmosphere_cruise_altitude(self):
        """Test ISA atmosphere at cruise altitude."""
        rho, a = get_atmosphere_properties(35000)  # FL350
        
        # At high altitude: lower density, lower speed of sound
        assert rho < 0.5  # Much less than sea level
        assert a < 320  # Lower than sea level
    
    def test_mach_number_calculation(self):
        """Test Mach number calculation."""
        # At FL350, 450 knots should be around M0.8
        mach = calculate_mach_number(
            groundspeed_kt=np.array([450.0]),
            altitude_ft=np.array([35000.0])
        )
        
        assert 0.75 < mach[0] < 0.85


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
