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
    RESOURCE_DIR: str = os.path.join(PROJECT_DIR, 'res')
    JOB_DIR: str = os.path.join(RESOURCE_DIR, 'trained_model')
    MODELS_DIR: str = os.path.join(JOB_DIR, 'models')
    CKPT_DIR: str = os.path.join(RESOURCE_DIR, 'ckpt')


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
