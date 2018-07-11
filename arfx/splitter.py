# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""Reorganize a long recording into a single file

This script collects data from a recording, possibly made over multiple ARF
files and splits it into chunks in a new file. Each new entry is
given an updated timestamp and attributes from the source entries.

Currently, no effort is made to splice data across entries or files. This may
result in some short entries. Also, only sampled datasets are processed.

"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import operator
import itertools
import logging
import arf
import h5py as h5

log = logging.getLogger('arfx-split')   # root logger


def entry_timestamps(arf_file):
    """Iterate through entries in arf file, yielding a seq of (entry, timestamp) tuples """
    for entry_name, entry in arf_file.items():
        if not isinstance(entry, h5.Group):
            continue
        entry_time = arf.timestamp_to_datetime(entry.attrs["timestamp"])
        yield (entry, entry_time)


def get_chunks(arf_file, dset_name, duration):
    """Iterate through entries in arf_file, identifying chunks of specified duration (in s)

    Yields a sequence of dicts, each with {timestamp, entry, offset}
    """
    import datetime

    for entry_name, entry in arf_file.items():
        if not isinstance(entry, h5.Group):
            continue
        entry_time = arf.timestamp_to_datetime(entry.attrs["timestamp"])
        dset = entry[dset_name]
        if not arf.is_time_series(dset):
            continue
        samples = dset.shape[0]
        sampling_rate = dset.attrs['sampling_rate']
        chunk_size = int(duration * sampling_rate)
        chunk_step = datetime.timedelta(seconds=float(chunk_size / sampling_rate))
        for i, offset in enumerate(range(0, samples, chunk_size)):
            yield {"timestamp": entry_time + chunk_step * i,
                   "entry": entry,
                   "offset": offset}


def main(argv=None):
    import argparse
    from .core import __version__

    p = argparse.ArgumentParser(prog="arfx-split", description=__doc__)
    p.add_argument('--version', action='version',
                   version='%(prog)s ' + __version__)
    p.add_argument('-v', help='verbose output', action='store_true', dest='verbose')

    p.add_argument("--duration", "-T", help="the maximum duration of entries "
                   "(default: %(default).2f seconds)", type=float, default=600)
    p.add_argument("--dset", "-d", help="the dataset name to analyze. default is to use the first "
                   "dataset in the first entry")
    p.add_argument("--compression", "-z", help="set compression level in output file "
                   "(default: %(default)d)", type=int, default=1)
    p.add_argument("src", help="the ARF files to analyze", nargs="+")
    p.add_argument("tgt", help="the ARF file to create (will overwrite!)")

    args = p.parse_args(argv)

    ch = logging.StreamHandler()
    formatter = logging.Formatter("[%(name)s] %(message)s")
    if args.verbose:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO
    log.setLevel(loglevel)
    ch.setLevel(loglevel)  # change
    ch.setFormatter(formatter)
    log.addHandler(ch)

    # open all input files and sort entries by timestamp
    log.info("sorting source file entries by timestamp")
    srcs = [h5.File(fname, "r") for fname in args.src]
    entries = sorted(itertools.chain.from_iterable(entry_timestamps(fp) for fp in srcs),
                     key=operator.itemgetter(0))

    # open output file

    # iterate through source entries, then chunk up datasets
    for entry, timestamp in entries:
        log.debug("source entry: %s%s", os.path.basename(entry.file.filename), entry.name)
