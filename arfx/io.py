# -*- mode: python -*-
"""
Provides read and write access to data for import/export to ARF. This is based
on a plugin architecture.

Copyright (C) 2011 Daniel Meliza <dmeliza@dylan.uchicago.edu>
Created 2011-09-19
"""

from importlib.metadata import entry_points
from pathlib import Path

_entrypoint = "arfx.io"


def open(filename: str | Path, *args, **kwargs):
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
    # only the lookup is guarded. Wrapping the handler's construction too meant
    # a ValueError it raised for its own reasons -- an invalid mode, say -- was
    # reported as there being no handler for the extension, which is both wrong
    # and the opposite of the actual problem
    try:
        (ep,) = entry_points(group=_entrypoint, name=ext)
    except ValueError:
        raise ValueError(f"No handler defined for files of type '{ext}'") from None
    return ep.load()(filename, *args, **kwargs)


def list_plugins() -> list[str]:
    """Returns the names of plugins registered to the arfx.io entry point"""
    return [ep.name for ep in entry_points(group=_entrypoint)]


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


# Variables:
# End:
