# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
Provides read and write access to data for import/export to ARF. This is based
on a plugin architecture.

Copyright (C) 2011 Daniel Meliza <dmeliza@dylan.uchicago.edu>
Created 2011-09-19
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
_entrypoint = 'arfx.io'


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

# Variables:
# End:
