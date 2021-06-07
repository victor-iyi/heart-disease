import sys
import logging

from abc import ABCMeta
from enum import IntEnum
from pprint import PrettyPrinter
from logging.config import fileConfig
from typing import Any, Dict, Optional, TextIO

# Custom libraries.
from heart_disease.config import consts


##############################################################################
# +--------------------------------------------------------------------------+
# | Log: For logging and printing download progress, etc...
# +--------------------------------------------------------------------------+
##############################################################################
class Log(metaclass=ABCMeta):
    # File logger configuration. See `config/conts.py:LOGGER`.
    # f_config: Dict[str, str] = consts.LOGGER.ROOT
    f_config: Dict[str, str] = consts.LOGGER.HEART_DISEASE

    fileConfig(f_config['path'], disable_existing_loggers=True)
    _logger: logging.Logger = logging.getLogger(f_config['name'])

    # Log Levels.
    Level: IntEnum = IntEnum('Level', names={
        'CRITICAL': 50,
        'ERROR': 40,
        'WARNING': 30,
        'INFO': 20,
        'DEBUG': 10,
        'NOTSET': 0,
    })

    # Log Level.
    level: int = _logger.level

    @staticmethod
    def setLevel(level: int) -> None:
        Log._logger.setLevel(level=level)

    @staticmethod
    def getLogger() -> logging.Logger:
        return Log._logger

    @staticmethod
    def setLogger(logger_dict: Dict[str, str]) -> None:
        if 'name' in logger_dict.keys() and 'path' in logger_dict.keys():
            raise KeyError(
                '`logger_dict` does not have both `name` & `path` keys.'
            )

        fileConfig(logger_dict['path'], disable_existing_loggers=True)
        Log._logger = logging.getLogger(logger_dict['name'])

    @staticmethod
    def debug(*args: Any, **kwargs: Any) -> None:
        Log._logger.debug(*args, **kwargs)

    @staticmethod
    def info(*args: Any, **kwargs: Any) -> None:
        Log._logger.info(*args, **kwargs)

    @staticmethod
    def warn(*args: Any, **kwargs: Any) -> None:
        Log._logger.warning(*args, **kwargs)

    @staticmethod
    def error(*args: Any, **kwargs: Any) -> None:
        Log._logger.error(*args, **kwargs)

    @staticmethod
    def critical(*args: Any, **kwargs: Any) -> None:
        Log._logger.critical(*args, **kwargs)

    @staticmethod
    def exception(*args: Any, **kwargs: Any) -> None:
        Log._logger.exception(*args, **kwargs)

    @staticmethod
    def fatal(*args: Any, code: int = -1, **kwargs: Any) -> None:
        Log._logger.fatal(*args, **kwargs)
        exit(code)

    @staticmethod
    def log(*args: Any, **kwargs: Any) -> None:
        """Logging method avatar based on verbosity.

        Args:
            *args (Any): List of arguments to be printed.

        Keyword Args:
            verbose (int, optional): Defaults to 1. Verbosity level.

        Returns:
            None
        """

        # No logging if verbose is not 'on'.
        if not kwargs.pop('verbose', 1):
            return

        Log._logger.log(Log.level, *args, **kwargs)

    @staticmethod
    def pretty(args: Any, stream: Optional[TextIO] = None, indent: int = 1,
               width: int = 80, depth: int = 0, *, compact: bool = False) -> None:
        """Handle pretty printing operations onto a stream using a set of
            configured parameters.

        Args:
            args (Any): Structured arguments to be printed.
            stream (BinaryIO, optional): Defaults to `sys.stdout`. The
                desired output stream. Stream must be writable. If omitted
                (or false),
                the standard output stream available at construction will
                be used.
            indent (int, optional): Defaults to 1. Number of spaces to indent
                for each level of nesting.
            width (int, optional): Defaults to 80. Attempted maximum number of
                columns in the output.
            depth (int, optional): Defaults to None. The maximum depth to print
                out nested structures.
            compact (bool, optional): Defaults to False. If true, several items
                will be combined in one line.
        """
        printer = PrettyPrinter(
                stream=stream or sys.stdout, indent=indent,
                width=width, depth=None if depth == 0 else depth,
                compact=compact
        )
        printer.pprint(args)

    @staticmethod
    def progress(count: int, max_count: int) -> None:
        """Prints task progress *(in %)*.

        Args:
            count {int}: Current progress so far.
            max_count {int}: Total progress length.
        """

        # Percentage completion.
        pct_complete: float = count / max_count

        # Status-message. Note the \r which means the line should
        # overwrite itself.
        msg: str = f'\r- Progress: {pct_complete:.02%}'

        # Print it.
        sys.stdout.write(msg)
        sys.stdout.flush()
