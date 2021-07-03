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

# Built-in libraries.
import os.path
from abc import ABCMeta
from typing import Dict

# Custom libraries.
from heart_disease.config.config import Config

# Exported configurations.
__all__ = [
    'FS', 'LOGGER', 'SETUP',
]


##############################################################################
# +--------------------------------------------------------------------------+
# | FS: File System.
# +--------------------------------------------------------------------------+
##############################################################################
class FS(metaclass=ABCMeta):
    # Project name & absolute directory.
    PROJECT_DIR: str = os.path.dirname(
        os.path.dirname(os.path.dirname(__file__))
    )
    APP_NAME: str = os.path.basename(PROJECT_DIR)

    # Libraries & configuration folders.
    LIB_DIR: str = os.path.join(PROJECT_DIR, 'heart_disease')
    CONFIG_DIR: str = os.path.join(LIB_DIR, 'config')

    # Resources & data directories.
    DATA_DIR: str = os.path.join(PROJECT_DIR, 'data')
    SAVED_MODELS: str = os.path.join(DATA_DIR, 'trained_model')


##############################################################################
# +--------------------------------------------------------------------------+
# | Setup configuration constants.
# +--------------------------------------------------------------------------+
##############################################################################
class SETUP(metaclass=ABCMeta):
    # Global setup configuration.
    __global = Config.from_cfg(os.path.join(FS.CONFIG_DIR,
                                            'setup/global.cfg'))

    # Build mode/type.
    MODE: str = __global['config']['MODE']

    # Python version.
    PY_VERSION: str = __global['config']['PY_VERSION']


##############################################################################
# +--------------------------------------------------------------------------+
# | Logger: Logging configuration paths.
# +--------------------------------------------------------------------------+
##############################################################################
class LOGGER(metaclass=ABCMeta):
    # Root Logger:
    ROOT: Dict[str, str] = {
        'name': 'root',
        'path': os.path.join(FS.CONFIG_DIR, f'logger/{SETUP.MODE}.cfg')
    }

    # Library Logger:
    HEART_DISEASE: Dict[str, str] = {
        'name': 'heart_disease',
        'path': os.path.join(FS.CONFIG_DIR, f'logger/{SETUP.MODE}.cfg')
    }
