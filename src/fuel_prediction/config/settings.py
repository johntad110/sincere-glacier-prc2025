"""
Configuration management for fuel prediction.
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Config:
    """Configuration container for fuel prediction models."""
    
    data: Dict[str, Any] = field(default_factory=dict)
    gbm: Dict[str, Any] = field(default_factory=dict)
    lstm: Dict[str, Any] = field(default_factory=dict)
    stacking: Dict[str, Any] = field(default_factory=dict)
    logging: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            Config instance
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)
    
    @classmethod
    def from_default(cls) -> 'Config':
        """
        Load default configuration.
        
        Returns:
            Config instance with default values
        """
        default_path = Path(__file__).parent / 'default.yaml'
        return cls.from_yaml(str(default_path))
    
    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Get nested configuration value.
        
        Args:
            *keys: Sequence of keys to navigate (e.g., 'gbm', 'params', 'learning_rate')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        obj = self.__dict__
        for key in keys:
            if isinstance(obj, dict) and key in obj:
                obj = obj[key]
            else:
                return default
        return obj
    
    def update(self, section: str, updates: Dict[str, Any]) -> None:
        """
        Update configuration section.
        
        Args:
            section: Section name ('data', 'gbm', 'lstm', 'stacking', 'logging')
            updates: Dictionary of updates to apply
        """
        if hasattr(self, section):
            section_dict = getattr(self, section)
            if isinstance(section_dict, dict):
                section_dict.update(updates)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or use default.
    
    Args:
        config_path: Optional path to  custom config file
        
    Returns:
        Config instance
    """
    if config_path:
        return Config.from_yaml(config_path)
    return Config.from_default()
