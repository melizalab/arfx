#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
import sys
from setuptools import setup, find_packages, Extension
from distutils.command.build_ext import build_ext

if sys.version_info[:2] < (3, 6):
    raise RuntimeError("Python version 3.6 or greater required.")

from arfx import __version__

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

class BuildExt(build_ext):
    def build_extensions(self):
        import numpy
        compiler_settings = {'include_dirs': []}
        compiler_settings['include_dirs'].insert(0, "include")
        compiler_settings['include_dirs'].append(numpy.get_include())
        c_opts = []
        for ext in self.extensions:
            for k, v in compiler_settings.items():
                setattr(ext, k, v)
            ext.extra_compile_args.extend(c_opts)
        build_ext.build_extensions(self)


setup(
    name='arfx',
    version=__version__,
    description=short_desc,
    long_description=long_desc,
    classifiers=[x for x in cls_txt.split("\n") if x],
    author='Dan Meliza',
    author_email="dan@meliza.org",
    maintainer='Dan Meliza',
    maintainer_email="dan@meliza.org",
    url="https://github.com/melizalab/arfx",

    packages=find_packages(exclude=["*test*"]),
    ext_modules=[Extension('arfx.pcmseqio',
                           sources=['src/pcmseqio.c', 'src/pcmseq.c']),],

    cmdclass={'build_ext': BuildExt},

    entry_points={'arfx.io': ['.pcm = arfx.pcmio:pcmfile',
                              '.dat = arfx.pcmio:pcmfile',
                              '.wav = ewave:wavfile',
                              '.pcm_seq2 = arfx.pcmseqio:pseqfile',
                              '.pcm_seq = arfx.pcmseqio:pseqfile',
                              '.pcmseq2 = arfx.pcmseqio:pseqfile',
                              ],
                  'console_scripts': ['arfx = arfx.core:arfx',
                                      'arfx-migrate = arfx.migrate:migrate_script',
                                      'arfx-split = arfx.splitter:main',
                                      'arfx-select = arfx.select:main',
                                      'arfx-collect-sampled = arfx.collect:collect_sampled_script',
                  ],
                  },

    install_requires=["arf>=2.6", "ewave>=1.0.5"],
    test_suite='tests'
)
# Variables:
# End:
