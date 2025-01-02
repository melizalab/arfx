# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
arfx is a tool for copying data to and from ARF files. It is intended for simple
operations involving single channel datasets, though it can also be used to copy
entire entries between ARF files.
"""
try:
    from importlib.metadata import version

    __version__ = version("arfx")
except Exception:
    # If package is not installed (e.g. during development)
    __version__ = "unknown"
