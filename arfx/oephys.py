# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""Script for converting open-ephys data in the raw binary format to ARF

This interface is based on the (2020-09-29) specification at
https://open-ephys.atlassian.net/wiki/spaces/OEW/pages/166789121/Flat+binary+format

"""
import os
import numpy as np
import logging
import arf

from arfx import core

log = logging.getLogger("arfx-oephys")


class dataset(object):
    def __init__(self, base, structure):
        self.attrs = dict(**structure)
        self.name = self.attrs["folder_name"].strip("/").replace("/", "_")

    def __repr__(self):
        return "<open-ephys dataset '/%s'>" % self.name


class continuous_dset(dataset):
    """Represents a dataset from a continuous processor (i.e. sampled data)"""

    def __init__(self, base, structure):
        self.path = os.path.join(base, "continuous", structure["folder_name"])
        self.nchannels = structure.pop("num_channels")
        self.channel_attrs = structure.pop("channels")
        self.sampling_rate = structure.pop("sample_rate")
        if len(self.channel_attrs) != self.nchannels:
            log.warn(
                "warning: channel metadata count (%d) and data channel count (%d) don't match",
                len(self.channel_attrs),
                self.nchannels,
            )
        super().__init__(base, structure)

        timestamps = np.load(os.path.join(self.path, "timestamps.npy"), mmap_mode="r")
        self.offset = timestamps[0] / self.sampling_rate

        self.dtype = np.dtype("int16")
        datfile = os.path.join(self.path, "continuous.dat")
        self.fp = open(datfile, "rb")
        self.fp.seek(0, 2)
        size = self.fp.tell() // self.dtype.itemsize
        self.fp.seek(0, 0)
        if size % self.nchannels != 0:
            raise IOError(
                "size of file '%s' (%d) is not a multiple of the channel count (%d)"
                % (datfile, size, self.nchannels)
            )
        self.nsamples = size // self.nchannels
        log.debug(
            "- %s: array (%d, %d) @ %.1f/s",
            structure["folder_name"],
            self.nsamples,
            self.nchannels,
            self.sampling_rate,
        )

    @property
    def size(self):
        return self.nsamples * self.nchannels

    def iter_chunks(self, samples):
        """A generator that retrieves the dataset in chunks of nsamples x nchannels"""
        offset = 0
        while True:
            buf = self.fp.read(samples * self.nchannels * self.dtype.itemsize)
            if len(buf) == 0:
                return
            data = np.frombuffer(buf, dtype=self.dtype)
            data.shape = (data.size // self.nchannels, self.nchannels)
            yield offset, data
            offset += data.shape[0]


class spikes_dset(dataset):
    """Represents a dataset of spike times (not implemented)"""

    def __init__(self, base, structure):
        raise NotImplementedError("spikes datasets not yet supported")


class event_dset(dataset):
    """Represents a dataset from an event-type processor (i.e. marked point process)"""

    def __init__(self, base, structure):
        super().__init__(base, structure)
        self.attrs.pop("event_metadata", None)  # metadata is not kept
        path = os.path.join(base, "events", structure["folder_name"])
        timestamps = np.load(os.path.join(path, "timestamps.npy"))
        channels = np.load(os.path.join(path, "channels.npy"))
        self.sampling_rate = structure["sample_rate"]
        if structure["type"] == "string":
            messages = np.load(os.path.join(path, "text.npy"))
            self.data = np.rec.fromarrays(
                [timestamps, messages, channels], names=("start", "message", "channel")
            )
            self.units = ("samples", "", "")
        elif structure["type"] == "int16":
            events = np.load(os.path.join(path, "channel_states.npy"))
            self.data = np.rec.fromarrays([timestamps, events], names=("start", "ttl"))
            self.units = ("samples", "")
        else:
            raise NotImplementedError(
                "%s type event datasets not supported" % structure["type"]
            )
        log.debug(
            "- %s: array %s @ %.1f/s (compound type)",
            structure["folder_name"],
            self.data.shape,
            self.sampling_rate,
        )

    @property
    def size(self):
        return self.data.size


class recording(object):
    """Represents the contents of a single open-ephys recording session.

    Each recording session (`experimentN/recordingM`) in the open-ephys
    hierarchy corresponds to a single ARF entry. The interface attempts to mimic the
    h5py/ARF API as much as possible, with datasets accessed like dictionary elements.
    Loading data from disk is lazy.

    """

    def __init__(self, path):
        import json

        self.attrs = {}
        self.datasets = {}

        with open(os.path.join(path, "structure.oebin"), "r") as fp:
            structure = json.load(fp)

        for processor in structure.pop("continuous", ()):
            try:
                dset = continuous_dset(path, processor)
            except NotImplementedError as e:
                log.warn("- %s skipped: %s", processor["folder_name"], e)
            else:
                self.datasets[dset.name] = dset
        for processor in structure.pop("spikes", ()):
            try:
                dset = spikes_dset(path, processor)
            except NotImplementedError as e:
                log.warn("- %s skipped: %s", processor["folder_name"], e)
            else:
                self.datasets[dset.name] = dset
        for processor in structure.pop("events", ()):
            try:
                dset = event_dset(path, processor)
            except NotImplementedError as e:
                log.warn("- %s skipped: %s", processor["folder_name"], e)
            else:
                self.datasets[dset.name] = dset
        self.attrs.update(structure)


def script(argv=None):
    import argparse
    import glob
    import re
    import datetime
    from tqdm import tqdm

    p = argparse.ArgumentParser(
        prog="arfx-oephys",
        description="Copy open-ephys data from raw binary format to an arf file",
    )
    p.add_argument(
        "--version", action="version", version="%(prog)s " + core.__version__
    )
    p.add_argument("--arf-version", action="version", version=arf.version_info())
    p.add_argument(
        "--help-datatypes",
        help="print available datatypes and exit",
        action="version",
        version=arf.DataTypes._doc(),
    )

    p.add_argument(
        "-v", "--verbose", action="store_true", help="show verbose log messages"
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="load data but do not write anything to disk",
    )
    p.add_argument(
        "--progress",
        "-p",
        action="store_true",
        help="show a progress bar",
    )

    p.add_argument("--skip-empty", action="store_true", help="skip empty datasets")
    p.add_argument(
        "-n",
        help="name entries or files using %(metavar)s",
        metavar="TEMPLATE",
        dest="template",
    )
    p.add_argument(
        "--channel-list",
        "-C",
        type=argparse.FileType("r"),
        help="file with a list of continuous channels to include, one per line. "
        "All other continuous channels will be excluded.",
    )
    p.add_argument(
        "-k",
        metavar="KEY=VALUE",
        dest="attrs",
        action=core.ParseKeyVal,
        help="specify attributes of entries",
    )
    p.add_argument(
        "-T",
        default=arf.DataTypes.UNDEFINED,
        metavar="DATATYPE",
        dest="datatype",
        action=core.ParseDataType,
        help="specify data type for continuous data (see --help-datatypes)",
    )
    p.add_argument(
        "-z",
        default=1,
        dest="compress",
        type=int,
        help="set compression level in ARF (default: %(default)s)",
    )

    p.add_argument(
        "-f",
        required=True,
        metavar="FILE",
        dest="arffile",
        help="the destination ARF file (new data are appended)",
    )
    p.add_argument("path", nargs="+", help="the source open-ephys data directories")

    args = p.parse_args(argv)
    core.setup_log(log, args.verbose)

    ts_regex = re.compile(r"(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})")

    fp_args = {}
    if args.dry_run:
        log.info("DRY RUN")
        fp_args.update(driver="core", backing_store=False)
    if args.channel_list is not None:
        cont_channels = [ch.strip("\n") for ch in args.channel_list.readlines()]
    else:
        cont_channels = None

    with arf.open_file(args.arffile, "a", **fp_args) as fp:
        log.info("Opened '%s' for writing", args.arffile)
        for path in args.path:
            try:
                m = ts_regex.search(path)
                timestamp = datetime.datetime(*(int(v) for v in m.groups()))
            except (AttributeError, ValueError) as e:
                log.error("can't parse timestamp from directory name %s: %s", path, e)
                p.exit(-1)
            for structfile in glob.iglob(
                os.path.join(path, "**/structure.oebin"), recursive=True
            ):
                dir = os.path.dirname(structfile)
                log.info("Reading from '%s':", dir)
                rec = recording(dir)

                rec_name = dir.replace("/", "_")
                entry = arf.create_entry(
                    fp,
                    name=rec_name,
                    timestamp=timestamp,
                    entry_creator="org.meliza.arfx/arfx-oephys " + core.__version__,
                    **rec.attrs
                )
                if args.attrs is not None:
                    arf.set_attributes(entry, **args.attrs)
                log.info("- creating entry '%s'", rec_name)

                for name, dset in rec.datasets.items():
                    if args.skip_empty and dset.size == 0:
                        log.info("  - skipping empty dataset '%s'", name)
                    elif isinstance(dset, continuous_dset):
                        log.info("  - processing continuous dataset '%s'", dset.name)
                        # first generate the datasets
                        datasets = []
                        for idx, info in enumerate(dset.channel_attrs):
                            if (
                                cont_channels is not None
                                and info["channel_name"] not in cont_channels
                            ):
                                log.info(
                                    "     - skipping unrequested channel '%s'",
                                    info["channel_name"],
                                )
                                continue
                            log.info(
                                "     - creating dataset for channel '%s'",
                                info["channel_name"],
                            )
                            # create an empty dataset and fill it in chunks
                            tgt = entry.create_dataset(
                                name=info["channel_name"],
                                shape=(dset.nsamples,),
                                dtype=dset.dtype,
                                compression=args.compress,
                                chunks=True,
                            )
                            chunksize = tgt.chunks[0]
                            arf.set_attributes(
                                tgt,
                                datatype=args.datatype,
                                offset=dset.offset,
                                sampling_rate=dset.sampling_rate,
                                **info
                            )
                            datasets.append((idx, tgt))
                        if len(datasets) == 0:
                            continue
                        log.info(
                            "  - reading %d samples in chunks of %d",
                            dset.nsamples,
                            chunksize,
                        )
                        expected = int(dset.nsamples / chunksize)
                        for offset, chunk in tqdm(
                            dset.iter_chunks(chunksize), total=expected, unit="chunk"
                        ):
                            nsamples, _ = chunk.shape
                            for i, tgt in datasets:
                                tgt[offset : offset + nsamples] = chunk[:, i]

                    elif isinstance(dset, event_dset):
                        log.info("  - processing event dataset '%s'", dset.name)
                        arf.create_dataset(
                            entry,
                            name=dset.name,
                            data=dset.data,
                            datatype=arf.DataTypes.EVENT,
                            sampling_rate=dset.sampling_rate,
                            units=dset.units,
                            **dset.attrs
                        )


if __name__ == "__main__":
    script()
