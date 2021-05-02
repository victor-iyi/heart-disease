"""Configuration module.

Features:
 - File configuration
 - Logging configuration
 - Utility/Handy classes and functions for:
   - Downloading and extracting files and folders.
   - Loading configuration files .json, .csv, .cfg, .in, .yaml, etc...
"""

from heart_disease.config.config import Config
from heart_disease.config.consts import FS, LOGGER, SETUP
from heart_disease.config.util import Log

__all__ = [
    # Configuration utils.
    'Config',

    # File system configurations.
    'FS', 'SETUP', 'LOGGER',

    # Utility files.
    'Log',
]
