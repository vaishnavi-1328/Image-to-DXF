"""
Configuration loader for the image-to-DXF conversion system.
"""
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from copy import deepcopy


class Config:
    """Configuration container with dot-notation access and nested updates."""

    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize config from dictionary.

        Args:
            config_dict: Configuration dictionary
        """
        self._config = deepcopy(config_dict)

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key_path: Dot-separated path to config value (e.g., "edge_detection.canny_low_threshold")
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation.

        Args:
            key_path: Dot-separated path to config value
            value: Value to set
        """
        keys = key_path.split('.')
        config = self._config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value

    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with dictionary of values.

        Args:
            updates: Dictionary of updates (supports nested dicts or dot notation keys)
        """
        for key, value in updates.items():
            if '.' in key:
                # Dot notation key
                self.set(key, value)
            else:
                # Direct key
                if isinstance(value, dict) and key in self._config and isinstance(self._config[key], dict):
                    # Recursive update for nested dicts
                    self._config[key].update(value)
                else:
                    self._config[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to dictionary.

        Returns:
            Configuration dictionary
        """
        return deepcopy(self._config)

    def __getitem__(self, key: str) -> Any:
        """
        Get configuration section.

        Args:
            key: Configuration section key

        Returns:
            Configuration section
        """
        return self._config[key]

    def __contains__(self, key: str) -> bool:
        """
        Check if key exists in configuration.

        Args:
            key: Configuration key

        Returns:
            True if key exists
        """
        return key in self._config

    def __repr__(self) -> str:
        return f"Config({self._config})"


def load_config(config_path: Optional[Path] = None) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file (defaults to config/defaults.yaml)

    Returns:
        Config object

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    if config_path is None:
        # Default to config/defaults.yaml relative to project root
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent
        config_path = project_root / "config" / "defaults.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    if config_dict is None:
        config_dict = {}

    return Config(config_dict)


def merge_cli_args(config: Config, cli_args: Dict[str, Any]) -> Config:
    """
    Merge command-line arguments into configuration.

    Args:
        config: Base configuration
        cli_args: Dictionary of CLI arguments (with None values filtered out)

    Returns:
        Updated Config object
    """
    # Filter out None values
    updates = {k: v for k, v in cli_args.items() if v is not None}

    # Map common CLI argument names to config paths
    mapping = {
        'image_type': 'classification.override_type',
        'canny_low': 'edge_detection.canny_low_threshold',
        'canny_high': 'edge_detection.canny_high_threshold',
        'simplify_epsilon': 'vectorization.simplify_epsilon_mm',
        'min_area': 'contour_extraction.min_contour_area_pixels',
        'min_feature_size': 'validation.min_feature_size_mm',
        'dpi': 'scale.default_dpi',
        'scale': 'scale.pixels_per_mm',
        'log_level': 'logging.level',
        'preview': 'preview.generate_previews',
        'interactive': 'preview.interactive_mode',
    }

    # Apply mappings
    mapped_updates = {}
    for cli_key, config_key in mapping.items():
        if cli_key in updates:
            mapped_updates[config_key] = updates[cli_key]

    config.update(mapped_updates)
    return config


def validate_config(config: Config) -> None:
    """
    Validate configuration values.

    Args:
        config: Configuration to validate

    Raises:
        ValueError: If configuration is invalid
    """
    # Validate threshold values
    canny_low = config.get('edge_detection.canny_low_threshold', 50)
    canny_high = config.get('edge_detection.canny_high_threshold', 150)

    if canny_low >= canny_high:
        raise ValueError(f"canny_low_threshold ({canny_low}) must be < canny_high_threshold ({canny_high})")

    if canny_low < 0 or canny_high < 0:
        raise ValueError("Canny thresholds must be non-negative")

    # Validate minimum area
    min_area = config.get('contour_extraction.min_contour_area_pixels', 100)
    if min_area < 0:
        raise ValueError("min_contour_area_pixels must be non-negative")

    # Validate DXF units
    units = config.get('dxf_output.units', 4)
    valid_units = [0, 1, 2, 3, 4, 5, 6]  # Common DXF unit codes
    if units not in valid_units:
        raise ValueError(f"dxf_output.units must be one of {valid_units} (4=mm is recommended)")

    # Validate DXF version
    version = config.get('dxf_output.version', 'R2010')
    valid_versions = ['R12', 'R2000', 'R2004', 'R2007', 'R2010', 'R2013', 'R2018']
    if version not in valid_versions:
        raise ValueError(f"dxf_output.version must be one of {valid_versions}")

    # Validate simplification epsilon
    epsilon = config.get('vectorization.simplify_epsilon_mm', 0.05)
    if epsilon < 0:
        raise ValueError("simplify_epsilon_mm must be non-negative")

    # Validate min feature size
    min_size = config.get('validation.min_feature_size_mm', 1.0)
    if min_size < 0:
        raise ValueError("min_feature_size_mm must be non-negative")
