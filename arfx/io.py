# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
Provides read and write access to data for import/export to ARF. This is based
on a plugin architecture.

Copyright (C) 2011 Daniel Meliza <dmeliza@dylan.uchicago.edu>
Created 2011-09-19
"""
from importlib.metadata import entry_points
from pathlib import Path
from typing import Optional, Type, Union

_entrypoint = "arfx.io"


def open(filename: Union[str, Path], *args, **kwargs):
    """Open a file and return an appropriate object, based on extension.

    The handler class is dynamically dispatched using Python's entry points system.
    Arguments are passed to the initializer for the handler.

    Args:
        filename: Path to the file to open
        *args: Positional arguments passed to the handler
        **kwargs: Keyword arguments passed to the handler

    Returns:
        An instance of the appropriate handler class

    Raises:
        ValueError: If no handler is found for the file extension

    """
    ext = Path(filename).suffix.lower()
    try:
        (ep,) = entry_points(group=_entrypoint, name=ext)
        cls = ep.load()
        return cls(filename, *args, **kwargs)
    except ValueError:
        raise ValueError(f"No handler defined for files of type '{ext}'")
    except TypeError:
        # shim for python < 3.10
        for ep in entry_points().get(_entrypoint, []):
            if ep.name == ext:
                cls = ep.load()
                return cls(filename, *args, **kwargs)
        raise ValueError(f"No handler defined for files of type '{ext}'")


def list_plugins() -> str:
    """Returns a string listing plugins registered to the arfx.io entry point"""
    try:
        eps = entry_points(group=_entrypoint)
    except TypeError:
        eps = entry_points().get(_entrypoint, [])
    return [ep.name for ep in eps]


def is_appendable(shape1, shape2):
    """Returns true if two array shapes are the same except for the first dimension"""
    from itertools import zip_longest

    return all(
        a == b
        for i, (a, b) in enumerate(zip_longest(shape1, shape2, fillvalue=1))
        if i > 0
    )


def extended_shape(shape1, shape2):
    """Returns the shape that results if two arrays are appended along the first dimension"""
    from itertools import zip_longest

    for i, (a, b) in enumerate(zip_longest(shape1, shape2, fillvalue=1)):
        if i == 0:
            yield a + b
        elif a == b:
            yield a
        else:
            raise ValueError(
                "data shape is not compatible with previously written data"
            )


def _get_handler_class(extension: str) -> Optional[Type]:
    """Get the (first) handler class for a given file extension.

    Args:
        extension: The file extension including the dot (e.g., '.txt')

    Returns:
        The handler class if found, None otherwise
    """
    # entry_points() accepts a group parameter in newer versions
    eps = entry_points()
    if hasattr(eps, "select"):  # Python 3.10+ style
        matching_eps = eps.select(group=_entrypoint, name=extension)
    else:  # Older style
        matching_eps = [ep for ep in eps.get(_entrypoint, []) if ep.name == extension]

    return matching_eps[0].load() if matching_eps else None


# Variables:
# End:
