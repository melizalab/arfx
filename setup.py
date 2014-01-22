#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
import sys
if sys.hexversion < 0x02060000:
    raise RuntimeError("Python 2.6 or higher required")

# setuptools 0.7+ doesn't play nice with distribute, so try to use existing
# package if possible
try:
    from setuptools import setup, find_packages, Extension
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()
    from setuptools import setup, find_packages, Extension

try:
    from Cython.Distutils import build_ext
    SUFFIX = '.pyx'
except ImportError:
    from distutils.command.build_ext import build_ext
    SUFFIX = '.c'

import sys
import numpy

# --- Distutils setup and metadata --------------------------------------------

VERSION = '2.2.2'

cls_txt = """
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
License :: OSI Approved :: GNU General Public License (GPL)
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Scientific/Engineering
Operating System :: Unix
Operating System :: POSIX :: Linux
Operating System :: MacOS :: MacOS X
Natural Language :: English
"""

short_desc = "Advanced Recording Format Tools"

long_desc = """Commandline tools for reading and writing Advanced Recording Format files.
ARF files are HDF5 files used to store audio and neurophysiological recordings
in a rational, hierarchical format. Data are organized around the concept of an
entry, which is a set of data channels that all start at the same time.

"""

compiler_settings = {
    'include_dirs': [numpy.get_include()],
}
if sys.platform == 'darwin':
    compiler_settings['include_dirs'] += ['/opt/local/include']

requirements = ["arf>=2.2", "ewave>=1.0.4"]
if sys.hexversion < 0x02070000:
    requirements.append("argparse==1.2.1")

setup(
    name='arfx',
    version=VERSION,
    description=short_desc,
    long_description=long_desc,
    classifiers=[x for x in cls_txt.split("\n") if x],
    author='Dan Meliza',
    author_email='"dan" at the domain "meliza.org"',
    maintainer='Dan Meliza',
    maintainer_email='"dan" at the domain "meliza.org"',
    url="https://github.com/dmeliza/arfx",

    packages=find_packages(exclude=["*test*"]),
    ext_modules=[Extension('arfx.pcmseqio',
                           sources=['src/pcmseqio.c', 'src/pcmseq.c'],
                           **compiler_settings),
                 Extension('arfx.h5vlen',
                           sources=['src/h5vlen' + SUFFIX],
                           libraries=['hdf5'],
                           **compiler_settings)],
    cmdclass={'build_ext': build_ext},
    entry_points={'arfx.io': ['.pcm = arfx.pcmio:pcmfile',
                              '.wav = ewave:wavfile',
                              '.pcm_seq2 = arfx.pcmseqio:pseqfile',
                              '.pcm_seq = arfx.pcmseqio:pseqfile',
                              '.pcmseq2 = arfx.pcmseqio:pseqfile',
                              ],
                  'console_scripts': ['arfx = arfx.arfx:arfx',
                                      'arfxplog = arfx.arfxplog:arfxplog'],
                  },

    install_requires=requirements,
    test_suite='nose.collector'
)
# Variables:
# End:
