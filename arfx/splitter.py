# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""This script collects data from a recording, possibly made over multiple ARF
files and splits it into chunks in a new file. Each new entry is
given an updated timestamp and attributes from the source entries.

Currently, no effort is made to splice data across entries or files. This may
result in some short entries. Also, only sampled datasets are processed.

"""
import argparse
import datetime
import itertools
import logging
import operator
from pathlib import Path
from typing import Iterator, Sequence, Tuple

import arf
import h5py as h5
import numpy as np

log = logging.getLogger("arfx-split")  # root logger


def entry_timestamps(
    arf_file: h5.Group,
) -> Iterator[Tuple[h5.Group, datetime.datetime]]:
    """Iterate through entries in arf file, yielding a seq of (entry, timestamp) tuples"""
    for _entry_name, entry in arf_file.items():
        if not isinstance(entry, h5.Group):
            continue
        entry_time = arf.timestamp_to_datetime(entry.attrs["timestamp"])
        yield (entry, entry_time)


def entry_duration(entry: h5.Group) -> float:
    """Determines the entry duration (in s) by finding the longest (sampled) dataset"""
    max_dur = 0
    for dset in entry.values():
        if not arf.is_time_series(dset):
            continue
        samples = dset.shape[0]
        sampling_rate = dset.attrs["sampling_rate"]
        max_dur = max(max_dur, float(samples / sampling_rate))
    return max_dur


def merge_jill_logs(files: Sequence[h5.Group]) -> np.ndarray:
    """Merge all the 'jill_log' datasets in files into a single structured record array"""

    out = [fp["jill_log"] for fp in files if "jill_log" in fp]
    if len(out) > 0:
        arr = np.concatenate(out)
        arr.sort(order=("sec", "usec"))
        return pad_log_messages(arr)


def pad_log_messages(dset: np.ndarray) -> np.ndarray:
    """Turn variable-length messages into fixed-length so h5py will store them"""
    if "message" not in dset.dtype.fields:
        raise ValueError("input must be a structured array with a 'message' field")
    min_length = max(len(s) for s in dset["message"])
    new_dtype = [(k, v) for k, (v, _) in dset.dtype.fields.items() if k != "message"]
    new_dtype.append(("message", h5.string_dtype(length=min_length)))
    return dset.astype(np.dtype(new_dtype))


def main(argv=None):

    from .core import __version__, setup_log

    p = argparse.ArgumentParser(prog="arfx-split", description=__doc__)
    p.add_argument("--version", action="version", version="%(prog)s " + __version__)
    p.add_argument("-v", help="verbose output", action="store_true", dest="verbose")

    p.add_argument(
        "--duration",
        "-T",
        help="the maximum duration of entries " "(default: %(default).2f seconds)",
        type=float,
        default=600,
    )
    p.add_argument(
        "--compress",
        "-z",
        help="set compression level in output file " "(default: %(default)d)",
        type=int,
        default=1,
    )
    p.add_argument(
        "--dry-run",
        "-n",
        help="don't actually create the target file or copy data",
        action="store_true",
    )
    p.add_argument(
        "--append",
        "-a",
        help="if true, will append data from src to tgt (default "
        "is to overwrite). Note that log files are NOT merged in this mode",
        action="store_true",
    )
    p.add_argument("src", type=Path, help="the ARF files to chunk up", nargs="+")
    p.add_argument("tgt", type=Path, help="the destination ARF file")

    args = p.parse_args(argv)
    setup_log(log, args.verbose)

    # open all input files and sort entries by timestamp
    log.info("sorting source file entries by timestamp")
    srcs = [h5.File(fname, "r") for fname in args.src]
    entries = sorted(
        itertools.chain.from_iterable(entry_timestamps(fp) for fp in srcs),
        key=operator.itemgetter(1),
    )
    if args.verbose:
        log.debug("entry order:")
        for entry, timestamp in entries:
            log.debug(
                "  %s%s (time=%s)",
                Path(entry.file.filename).name,
                entry.name,
                timestamp,
            )

    # open output file
    tgt_entry_index = 0
    if not args.dry_run:
        if args.append:
            tgt_file = arf.open_file(args.tgt, mode="a")
            log.info("appending to destination file: %s", tgt_file.filename)
            log.info("  counting entries...")
            tgt_entry_index = arf.count_children(tgt_file, h5.Group)
        else:
            tgt_file = arf.open_file(args.tgt, mode="w")
            log.info("created destination file: %s", tgt_file.filename)
            jilllog = merge_jill_logs(srcs)
            if jilllog is not None:
                tgt_file.create_dataset(
                    "jill_log", data=jilllog, compression=args.compress
                )
                log.info("merged jill_log datasets")

    # iterate through source entries, then chunk up datasets
    for entry, timestamp in entries:
        log.info("source entry: %s%s", Path(entry.file.filename).name, entry.name)
        max_duration = entry_duration(entry)
        n_chunks = int(max_duration // args.duration) + 1
        log.debug("  max duration: %3.2f s (chunks=%d)", max_duration, n_chunks)
        for i in range(n_chunks):
            tgt_entry_name = f"entry_{tgt_entry_index:05}"
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
                tgt_entry.attrs["origin-file"] = Path(entry.file.filename).name
                tgt_entry.attrs["origin-entry"] = Path(entry.name).name
            for dset_name, dset in entry.items():
                if not arf.is_time_series(dset):
                    log.debug("    %s: (not sampled)", dset_name)
                    continue
                sampling_rate = dset.attrs["sampling_rate"]
                chunk_size = int(args.duration * sampling_rate)
                start = chunk_size * i
                stop = min(start + chunk_size, dset.shape[0])
                log.debug("    %s: [%d:%d]", dset_name, start, stop)
                # data = dset[start:stop]
                if not args.dry_run:
                    tgt_attrs = dict(dset.attrs)
                    try:
                        tgt_attrs["origin-uuid"] = tgt_attrs.pop("uuid")
                    except KeyError:
                        pass
                    arf.create_dataset(
                        tgt_entry,
                        dset_name,
                        dset[start:stop],
                        compression=args.compress,
                        **tgt_attrs,
                    )
