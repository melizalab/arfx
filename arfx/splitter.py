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
import datetime
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


def entry_duration(entry):
    """Determines the entry duration by finding the longest (sampled) dataset"""
    max_dur = 0
    for dset in entry.values():
        if not arf.is_time_series(dset):
            continue
        samples = dset.shape[0]
        sampling_rate = dset.attrs['sampling_rate']
        max_dur = max(max_dur, float(samples / sampling_rate))
    return max_dur


def merge_jill_logs(files):
    """Merge all the 'jill_log' datasets in files"""
    from numpy import concatenate
    out = [fp['jill_log'] for fp in files if 'jill_log' in fp]
    if len(out) > 0:
        arr = concatenate(out)
        arr.sort(order=('sec', 'usec'))
        return arr


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
    p.add_argument("--compress", "-z", help="set compression level in output file "
                   "(default: %(default)d)", type=int, default=1)
    p.add_argument("--dry-run", "-n", help="don't actually create the target file or copy data",
                   action="store_true")
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
    tgt_entry_index = 0
    if not args.dry_run:
        tgt_file = arf.open_file(args.tgt, mode="w")
        log.info("created destination file: %s", tgt_file.filename)

    # merge log entries
    jilllog = merge_jill_logs(srcs)
    if jilllog is not None:
        tgt_file.create_dataset("jill_log", data=jilllog, compression=args.compress)
        log.info("merged jill_log datasets")

    # iterate through source entries, then chunk up datasets
    for entry, timestamp in entries:
        log.info("source entry: %s%s", os.path.basename(entry.file.filename), entry.name)
        max_duration = entry_duration(entry)
        n_chunks = int(max_duration // args.duration) + 1
        log.debug("  max duration: %3.2f s (chunks=%d)", max_duration, n_chunks)
        for i in range(n_chunks):
            tgt_entry_name = "entry_%05d" % tgt_entry_index
            tgt_timestamp = timestamp + datetime.timedelta(seconds=args.duration) * i
            # create target entry
            log.info("  target entry: %s (time=%s)", tgt_entry_name, tgt_timestamp)
            tgt_entry_index += 1
            # set target entry attributes
            if not args.dry_run:
                tgt_entry = arf.create_entry(tgt_file, tgt_entry_name, tgt_timestamp)
                for k, v in entry.attrs.items():
                    if k == "timestamp":
                        continue
                    elif k == "uuid":
                        k = "origin-uuid"
                    tgt_entry.attrs[k] = v
                tgt_entry.attrs["origin-file"] = os.path.basename(entry.file.filename)
                tgt_entry.attrs["origin-entry"] = os.path.basename(entry.name)
            for dset_name, dset in entry.items():
                if not arf.is_time_series(dset):
                    log.debug("    %s: (not sampled)", dset_name)
                    continue
                sampling_rate = dset.attrs['sampling_rate']
                chunk_size = int(args.duration * sampling_rate)
                start = chunk_size * i
                stop = min(start + chunk_size, dset.shape[0])
                data = dset[start:stop]
                log.debug("    %s: [%d:%d]", dset_name, start, stop)
                if not args.dry_run:
                    tgt_attrs = dict(dset.attrs)
                    try:
                        tgt_attrs['origin-uuid'] = tgt_attrs.pop('uuid')
                    except KeyError:
                        pass
                    arf.create_dataset(tgt_entry, dset_name, data, compression=args.compress,
                                       **tgt_attrs)
