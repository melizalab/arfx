# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
Specialized script to select and extract specific time segments into a new arf file.

The segments to extract are specified by a simple json structure:

{"entry": "name" or index, "begin": start_time, "end": stop_time"}

Multiple segments can be specified as a line-delimited json stream

Copyright (C) 2019 Dan Meliza <dan // AT // meliza.org>
"""
import sys
import logging
import json
import arf

log = logging.getLogger("arfx-collect")


def main(argv=None):
    import argparse
    from arfx.core import __version__, setup_log

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
    arf.check_file_version(src)

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
            tgt_entry_name = "entry_%05d" % tgt_entry_index
            log.info(
                " - %s: [%s, %s) -> %s",
                entry_name,
                interval["begin"],
                interval["end"],
                tgt_entry_name,
            )
            tgt_entry = arf.create_entry(tgt_file, tgt_entry_name, **src_entry_attrs)
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
                        src_units = src_dset_attrs.pop("units", "")
                        req = len(src_dset.dtype.names)
                        if isinstance(src_units, str) or len(src_units) != req:
                            src_dset_attrs["units"] = [src_units] + [""] * (req - 1)
                selected, offset = arf.select_interval(
                    src_dset, interval["begin"], interval["end"]
                )
                # this is to deal with jrecord-generated files that violate
                # spec on the units field
                arf.create_dataset(
                    tgt_entry,
                    name,
                    selected,
                    offset=offset + src_dset_offset,
                    **src_dset_attrs
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
