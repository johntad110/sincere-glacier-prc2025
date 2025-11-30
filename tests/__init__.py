"""Test package initialization."""

import pytest


def test_package_import():
    """Test that package can be imported."""
    import fuel_prediction
    assert fuel_prediction.__version__ == "1.0.0"
    assert fuel_prediction.__author__ == "Team Sincere Glacier"


def test_config_import():
    """Test configuration can be imported."""
    from fuel_prediction import load_config
    
    config = load_config()
    assert config is not None
    assert hasattr(config, 'gbm')
    assert hasattr(config, 'lstm')


def test_utils_import():
    """Test utilities can be imported."""
    from fuel_prediction import setup_logger, calculate_rmse
    
    assert callable(setup_logger)
    assert callable(calculate_rmse)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
