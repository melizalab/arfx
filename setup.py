#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
from distribute_setup import use_setuptools
use_setuptools()
from setuptools import setup, find_packages, Extension
try:
    from Cython.Distutils import build_ext
    SUFFIX = '.pyx'
except ImportError:
    from distutils.command.build_ext import build_ext
    SUFFIX = '.c'

import os,sys
import numpy

# --- Distutils setup and metadata --------------------------------------------

VERSION = '2.0.0-alpha'

cls_txt = \
"""
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
License :: OSI Approved :: GNU General Public License (GPL)
Programming Language :: Python
Programming Language :: C++
Programming Language :: MATLAB
Topic :: Scientific/Engineering
Operating System :: Unix
Operating System :: POSIX :: Linux
Operating System :: MacOS :: MacOS X
Natural Language :: English
"""

short_desc = "Advanced Recording Format Tools"

long_desc = \
"""
Commandline tools for reading and writing Advanced Recording Format files. ARF
files are HDF5 files used to store audio and neurophysiological recordings in a
rational, hierarchical format. Data are organized around the concept of an
entry, which is a set of data channels that all start at the same time.

Includes a commandline tool for importing and exporting data from ARF files.
"""

compiler_settings = {
    'libraries' : ['hdf5'],
    'include_dirs' : [numpy.get_include()],
    }
if sys.platform=='darwin':
    compiler_settings['include_dirs'] += ['/opt/local/include']


setup(
    name = 'arfx',
    version = VERSION,
    description = short_desc,
    long_description = long_desc,
    classifiers = [x for x in cls_txt.split("\n") if x],
    author = 'Dan Meliza',
    maintainer = 'Dan Meliza',
    maintainer_email = '"dan" at the domain "meliza.org"',
    url = "https://github.com/dmeliza/arfx",

    packages = find_packages(exclude=["*test*"]),
    ext_modules = [Extension('arfx._pcmseqio',
                             sources=['src/pcmseqio.c','src/pcmseq.c'], **compiler_settings),
                   Extension('arfx.h5vlen', sources=['src/h5vlen' + SUFFIX], **compiler_settings)],
    cmdclass = {'build_ext': build_ext},
    entry_points = {'arfx.io': [
            # '.pcm = arfx.io:pcmfile',
            '.wav = ewave:wavfile',
            # '.pcm_seq2 = arfx._pcmseqio:pseqfile',
            # '.pcm_seq = arfx._pcmseqio:pseqfile',
            # '.pcmseq2 = arfx._pcmseqio:pseqfile',
            # '.toe_lis = arfx.io:toefile',
            # '.toelis = arfx.io:toefile',
            # '.lbl = arfx.io:lblfile'
            ],
                    'console_scripts': ['arfx = arfx.arfx:arfx'],
                    },

    install_requires = ["distribute","arf>=2.0","ewave>=1.0","toelis>=1.0"],
    test_suite = 'nose.collector'
    )
# Variables:
# End:
