# -*- coding: utf-8 -*-
# -*- mode: python -*-
""" General programming tools """
import collections
import functools


class memoized(object):
    """Memoizing decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    instead of evaluating the function.
    """

    def __init__(self, func):
        self.func = func
        self.cache = {}
        functools.update_wrapper(self, func)

    def __call__(self, *args):
        if not isinstance(args, collections.Hashable):
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value

    def __repr__(self):
        """Return the function's docstring."""
        return self.func.__doc__

    def __get__(self, obj, objtype):
        """Support instance methods."""
        return functools.partial(self.__call__, obj)
