# config.py
import json
import os
import logging


class Config:
    _instance = None
    _config = None

    @classmethod
    def load(cls):
        """Load configuration from appsettings.json"""
        try:
            # Look for config file in different locations
            config_paths = [
                'appsettings.json',
                os.path.join(os.path.dirname(__file__), 'appsettings.json'),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'appsettings.json')
            ]

            # Try to load from any of the paths
            for path in config_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        cls._config = json.load(f)
                    logging.info(f"Config loaded from {path}")
                    return cls._config

            # If no config found, log warning and return None
            logging.warning("No config file found at: " + ", ".join(config_paths))
            return None

        except Exception as e:
            logging.error(f"Error loading config: {e}")
            return None

    @classmethod
    def get(cls, section, key=None):
        """Get configuration value"""
        if cls._config is None:
            cls._config = cls.load()

        if cls._config is None:
            raise ValueError("Configuration not loaded")

        if section not in cls._config:
            raise KeyError(f"Section '{section}' not found in config")

        if key is None:
            return cls._config[section]

        if key not in cls._config[section]:
            raise KeyError(f"Key '{key}' not found in section '{section}'")

        return cls._config[section][key]