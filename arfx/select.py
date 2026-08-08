# -*- mode: python -*-
"""
Specialized script to select and extract specific time segments into a new arf file.

The segments to extract are specified by a simple json structure:

{"entry": "name" or index, "begin": start_time, "end": stop_time"}

Multiple segments can be specified as a line-delimited json stream

Copyright (C) 2019 Dan Meliza <dan // AT // meliza.org>
"""

import json
import logging
import sys

import arf

log = logging.getLogger("arfx-select")


def main(argv=None):
    import argparse

    from arfx.core import __version__, check_file_version, setup_log

    p = argparse.ArgumentParser(prog="arfx-select", description=__doc__)
    p.add_argument("--version", action="version", version="%(prog)s " + __version__)
    p.add_argument(
        "-v", "--verbose", help="show verbose log messages", action="store_true"
    )
    p.add_argument(
        "-c",
        "--channels",
        help="list of channels to select (default all)",
        metavar="CHANNEL",
        nargs="+",
    )
    p.add_argument(
        "-y",
        "--dry-run",
        help="don't write the target data to disk",
        action="store_true",
    )
    p.add_argument(
        "-s",
        "--segments",
        help="load segments from file instead of stdin",
        type=open,
        default=sys.stdin,
    )
    p.add_argument(
        "--preserve-marked",
        help="copy marked point process datasets over without selecting",
        action="store_true",
    )

    p.add_argument("src", help="the input ARF file")
    p.add_argument("tgt", help="the output ARF file (will be overwritten)")

    args = p.parse_args(argv)
    setup_log(log, args.verbose)

    src = arf.open_file(args.src, "r")
    log.info("selecting from '%s'", args.src)
    check_file_version(src)

    entry_names = [n for n in arf.keys_by_creation(src) if arf.is_entry(src[n])]
    if args.dry_run:
        log.info("DRY RUN")
        tgt_file = arf.open_file(args.tgt, mode="w", driver="core", backing_store=False)
    else:
        log.info("writing to '%s'", args.tgt)
        tgt_file = arf.open_file(args.tgt, mode="w")

    tgt_entry_index = 0
    for line in args.segments:
        try:
            interval = json.loads(line)
            if isinstance(interval["entry"], int):
                entry_name = entry_names[interval["entry"]]
            else:
                entry_name = interval["entry"]
            src_entry = src[entry_name]
            src_entry_attrs = dict(src_entry.attrs)
            tgt_entry_name = f"entry_{tgt_entry_index:05d}"
            log.info(
                " - %s: [%s, %s) -> %s",
                entry_name,
                interval["begin"],
                interval["end"],
                tgt_entry_name,
            )
            if "timestamp" not in src_entry_attrs:
                # create_entry takes the timestamp positionally; passing it
                # through **attrs worked only because every conforming entry
                # has one, and raised a missing-argument TypeError otherwise
                log.error("    ! no timestamp on %s, skipping", entry_name)
                continue
            # the entry timestamp is not adjusted for the interval: the
            # dataset's offset attribute is what places it within the entry,
            # and that is set below
            timestamp = src_entry_attrs.pop("timestamp")
            # a selected interval is a new entry, not a copy of the source one.
            # Passing the uuid through gave every interval taken from the same
            # entry the same identity. Record where it came from instead.
            src_entry_attrs.pop("uuid", None)
            src_entry_attrs["source_entry"] = entry_name
            tgt_entry = arf.create_entry(
                tgt_file, tgt_entry_name, timestamp, **src_entry_attrs
            )
            for name, src_dset in src_entry.items():
                if args.channels is not None and name not in args.channels:
                    continue
                log.info("    - %s", name)
                src_dset_attrs = dict(src_dset.attrs)
                src_dset_offset = src_dset_attrs.pop("offset", 0)
                if arf.is_marked_pointproc(src_dset):
                    if args.preserve_marked:
                        tgt_file.copy(src_dset, tgt_entry, name=name)
                        continue
                    else:
                        # The spec requires one unit per compound field, and
                        # create_dataset enforces it. Older versions of jrecord
                        # write a single scalar for the whole record, which
                        # describes the 'start' field, so those have to be
                        # expanded before the dataset can be rewritten.
                        src_units = src_dset_attrs.get("units", "")
                        req = len(src_dset.dtype.names)
                        if isinstance(src_units, str | bytes):
                            src_units = [src_units]
                        if len(src_units) != req:
                            # pad (or truncate) to one unit per field. The
                            # filler has to match the string type already
                            # present: h5py refuses a mixed str/bytes sequence.
                            src_units = list(src_units)
                            filler = (
                                ""
                                if src_units and isinstance(src_units[0], str)
                                else b""
                            )
                            src_units = (src_units + [filler] * req)[:req]
                        # otherwise pass the attribute through untouched, which
                        # keeps its HDF5 type: rebuilding it as a list would
                        # rewrite a fixed-width string array as variable-length
                        src_dset_attrs["units"] = src_units
                selected, offset = arf.select_interval(
                    src_dset, interval["begin"], interval["end"]
                )
                arf.create_dataset(
                    tgt_entry,
                    name,
                    selected,
                    offset=offset + src_dset_offset,
                    **src_dset_attrs,
                )
            tgt_entry_index += 1

        except json.JSONDecodeError:
            log.error("invalid json: %s", line)
            continue
        except KeyError as e:
            log.error("%s", e)

    # copy top-level datasets and attributes
    for dset in src.values():
        if arf.is_entry(dset):
            continue
        tgt_file.copy(dset, tgt_file, name=dset.name)
    for k, v in src.attrs.items():
        if k not in tgt_file.attrs:
            tgt_file.attrs[k] = v
    return 0


if __name__ == "__main__":
    main()
