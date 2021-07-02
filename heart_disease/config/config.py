# Built-in libraries.
import os
import json
import pickle
import configparser

from abc import ABCMeta
from typing import Any, Callable, Dict, ForwardRef, NoReturn
from typing import Optional, Sequence, TypeVar, Union

# Third party libraries.
import yaml

# In order to use LibYAML bindings, which is much faster than pure Python.
# Download and install [LibYAML](https://pyyaml.org/wiki/LibYAML).
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

# Exported classes & functions.
__all__ = [
    'Config',
]

AttrKey = TypeVar('AttrKey', str, bytes)
AttrValue = Union[
    Dict[AttrKey, ForwardRef('AttrValue')],
    Sequence[ForwardRef('AttrValue')],
    int, float, str, bool
]


# noinspection PyUnresolvedReferences
class Attr(dict):
    """Get attributes.

    Examples:
        ```python
        >>> d = Attr({'foo':3})
        >>> d['foo']
        3
        >>> d.foo
        3
        >>> d.bar
        Traceback (most recent call last):
        ...
        AttributeError: 'Attr' object has no attribute 'bar'

        Works recursively

        >>> d = Attr({'foo':3, 'bar':{'x':1, 'y':2}})
        >>> isinstance(d.bar, dict)
        True
        >>> d.bar.x
        1

        Bullet-proof

        >>> Attr({})
        {}
        >>> Attr(d={})
        {}
        >>> Attr(None)
        {}
        >>> d = {'a': 1}
        >>> Attr(**d)
        {'a': 1}

        Set attributes

        >>> d = Attr()
        >>> d.foo = 3
        >>> d.foo
        3
        >>> d.bar = {'prop': 'value'}
        >>> d.bar.prop
        'value'
        >>> d
        {'foo': 3, 'bar': {'prop': 'value'}}
        >>> d.bar.prop = 'newer'
        >>> d.bar.prop
        'newer'


        Values extraction

        >>> d = Attr({'foo':0, 'bar':[{'x':1, 'y':2}, {'x':3, 'y':4}]})
        >>> isinstance(d.bar, list)
        True
        >>> from operator import attrgetter
        >>> map(attrgetter('x'), d.bar)
        [1, 3]
        >>> map(attrgetter('y'), d.bar)
        [2, 4]
        >>> d = Attr()
        >>> d.keys()
        []
        >>> d = Attr(foo=3, bar=dict(x=1, y=2))
        >>> d.foo
        3
        >>> d.bar.x
        1

        Still like a dict though

        >>> o = Attr({'clean':True})
        >>> o.items()
        [('clean', True)]

        And like a class

        >>> class Flower(Attr):
        ...     power = 1
        ...
        >>> f = Flower()
        >>> f.power
        1
        >>> f = Flower({'height': 12})
        >>> f.height
        12
        >>> f['power']
        1
        >>> sorted(f.keys())
        ['height', 'power']
        ```
    """

    def __init__(
        self,
        d: Optional[Dict[AttrKey, AttrValue]] = None,
        **kwargs: Any) -> None:
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name: AttrKey, value: AttrValue) -> None:
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                     if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(Attr, self).__setattr__(name, value)
        super(Attr, self).__setitem__(name, value)

    __setitem__ = __setattr__


#############################################################################
# +-------------------------------------------------------------------------+
# | Config: Configuration avatar class to convert save & load config files.
# +-------------------------------------------------------------------------+
#############################################################################
class Config(metaclass=ABCMeta):
    @staticmethod
    def from_yaml(file: str) -> Attr:
        """Load configuration from a YAML file.

        Args:
            file (str): A `.yml` or `.yaml` filename.

        Raises:
            AssertionError: File is not a YAML file.
            FileNotFoundError: `file` was not found.

        Returns:
            Attr: config dictionary object.
        """

        assert (file.endswith('yaml') or
                file.endswith('yml')), 'File is not a YAML file.'

        if not os.path.isfile(file):
            raise FileNotFoundError(f'{file} was not found')

        with open(file, mode='r') as f:
            cfg = Attr(yaml.load(f, Loader=Loader))

        return cfg

    @staticmethod
    def from_cfg(file: str, ext: str = 'cfg') -> Attr:
        """Load configuration from an cfg file.

        Args:
            file (str): An cfg filename.
            ext (str, optional): Defaults to 'cfg'. Config file extension.

        Raises:
            AssertionError: File is not an `${ext}` file.
            FileNotFoundError: `file` was not found.

        Returns:
            Attr: config dictionary object.
        """

        assert file.endswith(ext), f'File is not a/an `{ext}` file.'

        if not os.path.isfile(file):
            raise FileNotFoundError(f'{file} was not found')

        cfg = configparser.ConfigParser(dict_type=Attr)
        cfg.read(file)

        return cfg

    @staticmethod
    def from_json(file: str) -> Attr:
        """Load configuration from a json file.

        Args:
            file (str): A JSON filename.

        Raises:
            AssertionError: File is not a JSON file.
            FileNotFoundError: `file` was not found.

        Returns:
            Attr: config dictionary object.
        """

        assert file.endswith(('json', 'jsonld',
                              'json-ld')), 'File must be a JSON/JSON-LD file.'

        if not os.path.isfile(file):
            raise FileNotFoundError(f'{file} was not found')

        with open(file, mode='r') as f:
            cfg = Attr(json.load(f))

        return cfg

    @staticmethod
    def to_yaml(cfg: Attr, file: str, **kwargs: Any) -> None:
        """Save configuration object into a YAML file.

        Args:
            cfg (Attr): Configuration: as a dictionary instance.
            file (str): Path to write the configuration to.

        Keyword Args:
            Passed into `dumper`.

        Raises:
            AssertionError: `dumper` must be callable.
        """
        # Use LibYAML (which is much faster than pure Python) dumper.
        kwargs.setdefault('Dumper', Dumper)

        # Write to a YAML file.
        Config._to_file(cfg=cfg, file=file, dumper=yaml.dump, **kwargs)

    @staticmethod
    def to_json(cfg: Attr, file: str, **kwargs: Any) -> None:
        """Save configuration object into a JSON file.

        Args:
            cfg (Attr): Configuration: as dictionary instance.
            file (str): Path to write the configuration to.

        Keyword Args:
            Passed into `dumper`.

        Raises:
            AssertionError: `dumper` must be callable.
        """

        # Write to a JSON file.
        Config._to_file(cfg=cfg, file=file, dumper=json.dump, **kwargs)

    @staticmethod
    def to_cfg(cfg: Attr, file: str, **kwargs: Any) -> NoReturn:
        """Save configuration object into a cfg or ini file.

        Args:
            cfg (Any): Configuration: as dictionary instance.
            file (str): Path to write the configuration to.

        Keyword Args:
            Passed into `dumper`.
        """
        print(cfg, file, kwargs)
        raise NotImplementedError

    @staticmethod
    def to_pickle(cfg: Attr, file: str, **kwargs: Any) -> None:
        """Save configuration object into a pickle file.

        Args:
            cfg (Attr): Configuration: as dictionary instance.
            file (str): Path to write the configuration to.

        Keyword Args:
            Passed into `dumper`.

        Raises:
            AssertionError: `dumper` must be callable.
        """
        Config._to_file(cfg=cfg, file=file, dumper=pickle.dump, **kwargs)

    @staticmethod
    def _to_file(
        cfg: Attr,
        file: str,
        dumper: Callable[..., None],
        **kwargs: Any
    ) -> None:
        """Save configuration object into a file as allowed by `dumper`.

        Args:
            cfg (Attr): Configuration: as dictionary instance.
            file (str): Path to write the configuration to.
            dumper (Callable[[Config, TextIO, ...], None]): Function/callable
                handler to save object to disk.

        Keyword Args:
            Passed into `dumper`.

        Raises:
            AssertionError: `dumper` must be callable.
        """

        assert callable(dumper), '`dumper` must be callable.'

        # Create director(y|ies) if it doesn't exist.
        os.makedirs(file, exist_ok=True)

        # Write configuration to file.
        with open(file, mode='wb', encoding='UTF-8') as f:
            dumper(cfg, f, **kwargs)
