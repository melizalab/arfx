# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
Provides read and write access to data for import/export to ARF. This is based
on a plugin architecture.

Copyright (C) 2011 Daniel Meliza <dmeliza@dylan.uchicago.edu>
Created 2011-09-19
"""
from pkg_resources import iter_entry_points
_entrypointgroups = ('arfx.io',)


def load_handlers(entrypoints=_entrypointgroups):
    handlers = {}
    for epg in entrypoints:
        for ep in iter_entry_points(epg):
            try:
                handlers[ep.name.lower()] = ep.load()
            except:
                pass
    return handlers


_handlers = load_handlers()


def reload_handlers():
    global _handlers
    _handlers = load_handlers()


def extensions():
    """ Return list of extensions with registered plugins """
    return _handlers.keys()


def open(filename, *args, **kwargs):
    """
    Open a file and return an appropriate object, based on
    extension. The handler class is dynamically dispatched using the
    distribute plugin architecture (see package docstring for details)

    mode: mode to open the file. Some containers only support 'r' and 'w';
          others support 'a' and 'r+'
    addtional arguments are passed to the initializer for the handler
    """
    from os.path import splitext
    ext = splitext(filename)[1].lower()
    subclass = _handlers.get(ext, None)
    if subclass is None: raise TypeError, "No handler defined for files of type %s" % ext
    return subclass(filename, *args, **kwargs)

# Variables:
# End:
