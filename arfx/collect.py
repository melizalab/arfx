# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
Specialized script to collect data across channels and entries

Copyright (C) 2018 Dan Meliza <dan // AT // meliza.org>
"""
import os
import operator
import numpy as np
import logging
import arf

from .core import __version__, setup_log
from . import io

log = logging.getLogger("arfx-collect")


def any_type(dset):
    return True


def first(dict, fun):
    """For a nested dict, return the first value of fun(subdict)"""
    for v in dict.values():
        return fun(v)


def all_items_equal(dict, fun):
    """For a nested dict, returns True iff all values of subdict[key] are equal"""
    ss = set(fun(v) for v in dict.values())
    return len(ss) <= 1


def channel_properties(entry, channels=None, predicate=any_type):
    """Returns a dict with channel names and required channel properties"""
    return {
        name: {
            "sampling_rate": dset.attrs.get("sampling_rate", None),
            "units": dset.attrs.get("units", None),
            "dtype": dset.dtype,
            "samples": dset.shape[0],
            "channels": arf.count_channels(dset),
            "chunksize": dset.chunks[0],
        }
        for name, dset in entry.items()
        if predicate(dset) and (channels is None or name in channels)
    }


def check_entry_consistency(arfp, entries=None, channels=None, predicate=any_type):
    """Check whether all entries in arfp have the required channels

    Raises a warning if units and sampling rates do not match across channels.

    entries - if not None, restrict to entries with supplied names
    channels - if not None, only check datasets with supplied names
    filter - a predicate on dataset (e.g. arf.is_time_series)

    If consistent, returns
      [ [included entry names in order of creation alphabetically],
        {
         channel_name: {'samping_rate', 'units', 'channels'},
         ...
        }
      ]

    If not consistent across entries, logs an error and returns None. If
    sampling rate and units are not consistent within an entry, logs a warning.

    """
    import h5py as h5

    log.info("checking entry consistency")
    # FIXME catch error when file does not track creation order
    entry_names = []
    channel_props = None
    for entry_name in arf.keys_by_creation(arfp):
        if entries is not None and entry_name in entries:
            continue
        entry = arfp[entry_name]
        if not isinstance(entry, h5.Group):
            continue
        props = channel_properties(entry, channels, predicate)
        sample_counts = set(v.pop("samples") for v in props.values())
        if len(sample_counts) > 1:
            log.error("sample count differs across channels in entry %s", entry_name)
            return
        if channel_props is None:
            channel_props = props
        elif props != channel_props:
            log.error("channels in entry %s do not match", entry_name)
            return
        entry_names.append(entry_name)
    return entry_names, channel_props


def iter_entry_chunks(entry, channels, predicate):
    """Iterate through the datasets in entry (that match predicate), yielding chunks"""
    from tqdm import tqdm

    props = channel_properties(entry, channels, predicate)
    nchannels = len(props)
    nsamples = first(props, operator.itemgetter("samples"))
    dtype = first(props, operator.itemgetter("dtype"))
    size = first(props, operator.itemgetter("chunksize"))
    log.info("    - '%s' -> %d samples", entry.name, nsamples)
    for i in tqdm(range(0, nsamples, size), unit="chunk"):
        n = min(size, nsamples - i)
        out = np.empty((n, nchannels), dtype=dtype)
        log.debug("            - %d - %d", i, i + n)
        for j, chan_name in enumerate(props):
            dset = entry[chan_name]
            out[:, j] = dset[i : i + n]
        yield out


def collect_sampled_script(argv=None):
    from natsort import natsorted
    import argparse

    p = argparse.ArgumentParser(
        prog="arfx-collect-sampled",
        description="Collect sampled data from arf files across channels and entries"
        "into a flat binary array. The output file can be any format that supports multiple channels; "
        "for example, wav or dat (raw binary)",
    )
    p.add_argument("--version", action="version", version="%(prog)s " + __version__)
    p.add_argument(
        "-v", "--verbose", help="show verbose log messages", action="store_true"
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="check entry consistency but don't write file",
    )

    p.add_argument(
        "-d",
        "--dtype",
        help="convert data to specified type (default is to use as stored)",
    )

    p.add_argument(
        "-c",
        "--channels",
        metavar="CHANNEL",
        nargs="+",
        help="list of channels to unpack (default all)",
    )
    p.add_argument(
        "-C",
        "--channel-file",
        help="file with list of channels to unpack, one per line",
    )
    p.add_argument(
        "-e",
        "--entries",
        help="list of entries to unpack (default all)",
        metavar="ENTRY",
        nargs="+",
    )
    p.add_argument(
        "--start",
        type=int,
        help="if set, only collect data after this time point (in samples)",
    )
    p.add_argument(
        "--stop",
        type=int,
        help="if set, only collect data before this time point (in samples)",
    )
    p.add_argument(
        "--mountain-params",
        action="store_true",
        help="create mountainlab params.json file",
    )

    p.add_argument("arffile", help="the ARF file to unpack")
    p.add_argument("outfile", help="the output file (will be overwritten)")

    args = p.parse_args(argv)
    setup_log(log, args.verbose)

    if args.channel_file is not None:
        with open(args.channel_file, "rt") as fp:
            if args.channels is None:
                args.channels = []
            for line in fp:
                if line.startswith("#"):
                    continue
                else:
                    args.channels.append(line.strip())

    with arf.open_file(args.arffile, "r") as arfp:
        log.info("unpacking '%s'", args.arffile)
        arf.check_file_version(arfp)
        entry_names, channel_props = check_entry_consistency(
            arfp, args.entries, args.channels, predicate=arf.is_time_series
        )
        if not all_items_equal(channel_props, operator.itemgetter("sampling_rate")):
            log.warn(" - warning: not all datasets have the same sampling rate")
        if not all_items_equal(channel_props, operator.itemgetter("units")):
            log.warn(" - warning: not all datasets have the same units")
        nentries = len(entry_names)
        nchannels = sum(channel_props[c]["channels"] for c in channel_props)
        sampling_rate = first(channel_props, operator.itemgetter("sampling_rate"))
        if args.dtype is None:
            dtype = first(channel_props, operator.itemgetter("dtype"))
        else:
            dtype = args.dtype
        log.info(" - channels (%d):", nchannels)
        for cname in natsorted(channel_props):
            log.info("    - %s", cname)
        if args.mountain_params:
            import json

            path = os.path.join(os.path.dirname(args.outfile), "params.json")
            log.info("writing mountainlab metadata to '%s'", path)
            data = {"samplerate": int(sampling_rate), "spike_sign": -1}
            with open(path, "wt") as jfp:
                json.dump(data, jfp)
        log.info("opening '%s' for output", args.outfile)
        log.info(" - sampling rate = %f", sampling_rate)
        log.info(" - dtype = '%s'", dtype)
        log.info(" - entries (%d):", nentries)
        if args.dry_run:
            log.info(" - dry run, ending script")
            return
        sample_count = 0
        with io.open(
            args.outfile,
            mode="w",
            sampling_rate=sampling_rate,
            dtype=dtype,
            nchannels=nchannels,
        ) as ofp:
            for entry_name in natsorted(entry_names):
                entry = arfp[entry_name]
                for chunk in iter_entry_chunks(
                    entry, args.channels, arf.is_time_series
                ):
                    if args.start is None or sample_count > args.start:
                        ofp.write(chunk.astype(dtype))
                    sample_count += chunk.shape[0]
                    if args.stop is not None and sample_count > args.stop:
                        log.info(
                            " - stopping as requested after writing %d samples",
                            sample_count - (args.start or 0),
                        )
                        return
