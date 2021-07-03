# Copyright 2021 Victor I. Afolabi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
