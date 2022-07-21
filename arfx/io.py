# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
Provides read and write access to data for import/export to ARF. This is based
on a plugin architecture.

Copyright (C) 2011 Daniel Meliza <dmeliza@dylan.uchicago.edu>
Created 2011-09-19
"""
_entrypoint = "arfx.io"


def open(filename, *args, **kwargs):
    """Open a file and return an appropriate object, based on extension.

    The handler class is dynamically dispatched using setuptools plugin
    architecture. (see package docstring for details)

    arguments are passed to the initializer for the handler

    """
    from pkg_resources import iter_entry_points
    from os.path import splitext

    ext = splitext(filename)[1].lower()
    cls = None
    for ep in iter_entry_points(_entrypoint, ext):
        cls = ep.load()
    if cls is None:
        raise TypeError("No handler defined for files of type '%s'" % ext)
    return cls(filename, *args, **kwargs)


def list_plugins():
    """Returns a printable list of plugins registered to the arfx.io entry point"""
    from pkg_resources import iter_entry_points

    return "Supported file formats: " + " ".join(
        ep.name for ep in iter_entry_points(_entrypoint)
    )


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
