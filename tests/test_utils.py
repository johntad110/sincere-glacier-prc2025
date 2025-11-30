"""
Test suite for fuel prediction utilities.
"""
import pytest
import numpy as np
import pandas as pd

from fuel_prediction.utils.metrics import calculate_rmse, get_fold_statistics
from fuel_prediction.utils.constants import RARE_AC_MAP, GBM_FEATURES


class TestMetrics:
    """Test metric calculations."""
    
    def test_rmse_perfect_predictions(self):
        """Test RMSE with perfect predictions."""
        y_true = np.array([100, 200, 300, 400])
        y_pred = np.array([100, 200, 300, 400])
        
        rmse = calculate_rmse(y_true, y_pred)
        assert rmse == 0.0
    
    def test_rmse_known_value(self):
        """Test RMSE with known result."""
        y_true = np.array([3, -0.5, 2, 7])
        y_pred = np.array([2.5, 0.0, 2, 8])
        
        # Expected: sqrt(mean([0.25, 0.25, 0, 1])) = sqrt(0.375) ≈ 0.612
        rmse = calculate_rmse(y_true, y_pred)
        assert abs(rmse - 0.612) < 0.01
    
    def test_fold_statistics(self):
        """Test fold statistics calculation."""
        scores = [10.0, 12.0, 11.0, 13.0, 9.0]
        
        mean, std, min_val, max_val = get_fold_statistics(scores)
        
        assert mean == 11.0
        assert min_val == 9.0
        assert max_val == 13.0
        assert std > 0


class TestConstants:
    """Test constant definitions."""
    
    def test_rare_ac_map_exists(self):
        """Test aircraft mapping exists."""
        assert len(RARE_AC_MAP) > 0
        assert 'A388' in RARE_AC_MAP
        assert RARE_AC_MAP['A388'] == 'B744'
    
    def test_gbm_features_not_empty(self):
        """Test GBM feature list exists."""
        assert len(GBM_FEATURES) > 0
        assert 'duration' in GBM_FEATURES
        assert 'aircraft_type' in GBM_FEATURES


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
