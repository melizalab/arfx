# -*- mode: python -*-
"""Script for converting open-ephys data in the raw binary format to ARF

This interface is based on the specification at
https://open-ephys.atlassian.net/wiki/spaces/OEW/pages/166789121/Flat+binary+format (for old and new formats) and on direct inspection of the output from pre and post 0.6 versions of the open-ephys GUI

"""

import datetime
import logging
import re
from pathlib import Path

import arf
import numpy as np
from natsort import natsorted
from packaging.version import Version

from arfx import core

log = logging.getLogger("arfx-oephys")


def first_existing(*paths: Path) -> Path | None:
    """Iterate through a sequence of paths and return the first one that exists, or None if none do"""
    for path in paths:
        if path.exists():
            return path


class dataset:
    def __init__(self, base, structure):
        self.attrs = dict(**structure)
        self.name = self.attrs["folder_name"].strip("/").replace("/", "_")

    def __repr__(self):
        return f"<open-ephys dataset '/{self.name}'>"


class continuous_dset(dataset):
    """Represents a dataset from a continuous processor (i.e. sampled data)"""

    def __init__(self, base: Path, structure: dict, gui_version: Version):
        self.path = base / "continuous" / structure["folder_name"]
        self.nchannels = structure.pop("num_channels")
        self.channel_attrs = structure.pop("channels")
        self.sampling_rate = structure.pop("sample_rate")
        if len(self.channel_attrs) != self.nchannels:
            log.warning(
                "warning: channel metadata count (%d) and data channel count (%d) don't match",
                len(self.channel_attrs),
                self.nchannels,
            )
        super().__init__(base, structure)

        # Sample index file. This can get deleted during spike sorting, and is
        # empty for an aborted recording, so in either case we fall back on the
        # sync_messages.txt file rather than failing.
        if gui_version < Version("0.6.0"):
            tspath = self.path / "timestamps.npy"
        else:
            tspath = self.path / "sample_numbers.npy"
        sample_offset = None
        if not tspath.exists():
            log.warning(
                "- warning: %s file is missing for %s; falling back on sync_messages.txt",
                tspath.name,
                self.name,
            )
        else:
            timestamps = np.load(tspath, mmap_mode="r")
            if timestamps.size > 0:
                sample_offset = timestamps[0]
            else:
                log.warning(
                    "- warning: %s is empty for %s; falling back on sync_messages.txt",
                    tspath.name,
                    self.name,
                )
        if sample_offset is None:
            sample_offset = find_sync_time(base, structure, gui_version)
            if sample_offset is None:
                raise RuntimeError(
                    f"unable to determine sync time for {self.name} dataset"
                )

        self.offset = sample_offset / self.sampling_rate
        self.dtype = np.dtype("int16")
        datfile = self.path / "continuous.dat"
        self.fp = open(datfile, "rb")
        self.fp.seek(0, 2)
        size = self.fp.tell() // self.dtype.itemsize
        self.fp.seek(0, 0)
        if size % self.nchannels != 0:
            raise OSError(
                f"size of file '{datfile}' ({size:d}) is not a multiple of the channel count ({self.nchannels})"
            )
        self.nsamples = size // self.nchannels
        log.debug(
            "- %s: array (%d, %d) @ %.1f/s; offset=%.3f s",
            structure["folder_name"],
            self.nsamples,
            self.nchannels,
            self.sampling_rate,
            self.offset,
        )

    @property
    def size(self) -> int:
        return self.nsamples * self.nchannels

    def iter_chunks(self, nsamples: int):
        """A generator that retrieves the dataset in chunks of nsamples x nchannels"""
        offset = 0
        while True:
            buf = self.fp.read(nsamples * self.nchannels * self.dtype.itemsize)
            if len(buf) == 0:
                return
            data = np.frombuffer(buf, dtype=self.dtype)
            # reshape rather than assigning to .shape, which numpy 2.5
            # deprecated; both are views, so nothing is copied
            data = data.reshape(data.size // self.nchannels, self.nchannels)
            yield offset, data
            offset += data.shape[0]


class spikes_dset(dataset):
    """Represents a dataset of spike times (not implemented)"""

    def __init__(self, base: Path, structure: dict, gui_version: Version):
        raise NotImplementedError("spikes datasets not yet supported")


class event_dset(dataset):
    """Represents a dataset from an event-type processor (i.e. marked point process)"""

    def __init__(self, base: Path, structure: dict, gui_version: Version):
        super().__init__(base, structure)
        self.attrs.pop("event_metadata", None)  # metadata is not kept
        path = base / "events" / structure["folder_name"]
        # Event times (in samples)
        if gui_version < Version("0.6.0"):
            timestamps = np.load(path / "timestamps.npy")
        else:
            timestamps = np.load(path / "sample_numbers.npy")
        self.sampling_rate = structure["sample_rate"]
        if structure["type"] == "string":
            messages = np.load(path / "text.npy")
            if gui_version < Version("0.6.0"):
                # Not exactly sure what this file holds, but we'll read it
                # if it's there
                channels = np.load(path / "channels.npy")
                self.data = np.rec.fromarrays(
                    [timestamps, messages, channels],
                    names=("start", "message", "channel"),
                )
                self.units = ("samples", "", "")
            else:
                self.data = np.rec.fromarrays(
                    [timestamps, messages], names=("start", "message")
                )
                self.units = ("samples", "")
        elif structure["type"] == "int16":
            if gui_version < Version("0.6.0"):
                events = np.load(path / "channel_states.npy")
            else:
                events = np.load(path / "states.npy")
            self.data = np.rec.fromarrays([timestamps, events], names=("start", "ttl"))
            self.units = ("samples", "")
        else:
            raise NotImplementedError(
                f"{structure['type']} type event datasets not supported"
            )
        log.debug(
            "- %s: array %s @ %.1f/s (compound type)",
            structure["folder_name"],
            self.data.shape,
            self.sampling_rate,
        )

    @property
    def size(self) -> int:
        return self.data.size


class recording:
    """Represents the contents of a single open-ephys recording session.

    Each recording session (`experimentN/recordingM`) in the open-ephys
    hierarchy corresponds to a single ARF entry. The interface attempts to mimic
    the h5py/ARF API as much as possible, with datasets accessed like dictionary
    elements. Loading data from disk is lazy.

    """

    def __init__(self, path: Path):
        import json

        self.attrs = {}
        self.datasets = {}

        with open(path / "structure.oebin") as fp:
            structure = json.load(fp)

        # need to dispatch on version because the format layout changed
        gui_version = Version(structure["GUI version"])

        for processor in structure.pop("continuous", ()):
            try:
                dset = continuous_dset(path, processor, gui_version)
            except NotImplementedError as e:
                log.warning("- %s skipped: %s", processor["folder_name"], e)
            else:
                self.datasets[dset.name] = dset
        for processor in structure.pop("spikes", ()):
            try:
                dset = spikes_dset(path, processor, gui_version)
            except NotImplementedError as e:
                log.warning("- %s skipped: %s", processor["folder_name"], e)
            else:
                self.datasets[dset.name] = dset
        for processor in structure.pop("events", ()):
            try:
                dset = event_dset(path, processor, gui_version)
            except NotImplementedError as e:
                log.warning("- %s skipped: %s", processor["folder_name"], e)
            else:
                self.datasets[dset.name] = dset
        self.attrs.update(structure)


# sync_messages.txt is not covered by the format documentation, and its wording
# changed completely at GUI 0.6 along with the way a stream is identified: by a
# numeric sub-processor index before, by name after. The two share no literal
# text, so a reader that knows only one of them matches nothing in the other
# rather than failing in any visible way.
_SYNC_RX_LEGACY = re.compile(
    r"Processor: (?P<processor>.+?) Id: (?P<id>\d+?) "
    r"subProcessor: (?P<stream>\d+?) start time: (?P<start>\d+)"
)
_SYNC_RX = re.compile(
    r"Start Time for (?P<processor>.+?) \((?P<id>\d+)\) - (?P<stream>.+?) "
    r"@ [\d.]+ \w*Hz: (?P<start>\d+)"
)


# The same file also carries a wall-clock time for the recording, but only in
# the newer wording. The pre-0.6 line is "Software time: 153530@1000000Hz",
# which is a monotonic clock reading, not an epoch -- using it as a timestamp
# would put the recording in 1970. Matching on the wording rather than the
# version is what keeps them apart: only one of the two forms says what its
# epoch is.
_SOFTWARE_TIME_RX = re.compile(
    r"Software Time \(milliseconds[^)]*\):\s*(?P<ms>\d+)", re.IGNORECASE
)


def find_software_time(path: Path) -> datetime.datetime | None:
    """The wall-clock start of a recording, from sync_messages.txt.

    Returns None when the file says nothing usable, which is the case for every
    recording written before GUI 0.6. The caller falls back to the timestamp in
    the session directory name -- correct for the first recording of a session
    and increasingly wrong for the rest.
    """
    try:
        text = (path / "sync_messages.txt").read_text()
    except OSError:
        return None
    match = _SOFTWARE_TIME_RX.search(text)
    if match is None:
        return None
    return datetime.datetime.fromtimestamp(int(match.group("ms")) / 1000)


def find_sync_time(path: Path, structure: dict, gui_version: Version) -> int | None:
    """Look up the start sample for a processor in sync_messages.txt.

    This file is used as a fallback if the sample times file is missing. Takes
    the whole continuous-processor structure rather than individual fields,
    because which fields identify a stream depends on the GUI version. Returns
    None if there is no matching line.

    """
    if gui_version < Version("0.6.0"):
        rx = _SYNC_RX_LEGACY
        stream = structure.get("source_processor_sub_idx")

        # the legacy sub-processor index is numeric on both sides
        def same_stream(found):
            return int(found) == int(stream)
    else:
        rx = _SYNC_RX
        stream = structure.get("stream_name")

        def same_stream(found):
            return found == stream

    if stream is None:
        log.debug("- no stream identifier in structure.oebin for this GUI version")
        return None

    name = structure["source_processor_name"]
    ident = int(structure["source_processor_id"])
    with open(path / "sync_messages.txt") as fp:
        for line in fp:
            match = rx.match(line)
            if match is None:
                continue
            if (
                match.group("processor") == name
                and int(match.group("id")) == ident
                and same_stream(match.group("stream"))
            ):
                return int(match.group("start"))
    return None


def script(argv=None):
    import argparse

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
        version=core.format_list(),
    )

    p.add_argument(
        "-v", "--verbose", action="store_true", help="show verbose log messages"
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="load data but do not write anything to disk",
    )

    p.add_argument("--skip-empty", action="store_true", help="skip empty datasets")
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
        help=(
            "specify attributes of entries; the value is read as JSON when it parses (so pen=1 is a number), otherwise as a string"
        ),
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
    p.add_argument(
        "path", type=Path, nargs="+", help="the source open-ephys data directories"
    )

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
                m = ts_regex.search(str(path))
                session_time = datetime.datetime(*(int(v) for v in m.groups()))
            except (AttributeError, ValueError) as e:
                log.error("can't parse timestamp from directory name %s: %s", path, e)
                p.exit(-1)
            # natsorted, so experiment10 follows experiment9 rather than
            # experiment1, and the entries land in the order they were recorded
            for structfile in natsorted(path.glob("**/structure.oebin")):
                dir = structfile.parent
                log.info("Reading from '%s':", dir)
                rec = recording(dir)

                # A session directory can hold several experiments and several
                # recordings per experiment, and they do not all start when the
                # session does. Taking the timestamp from the directory name
                # gave every one of them the start of the first.
                timestamp = find_software_time(dir)
                if timestamp is None:
                    timestamp = session_time
                    log.debug(
                        "- no wall-clock time in sync_messages.txt, "
                        "using the session directory's"
                    )

                # Named from the session directory down, not from the whole
                # path: the name used to be str(dir) with the separators
                # replaced, so the same recording imported from a different
                # machine, mount point, or working directory produced a
                # different entry name for identical data, and two such
                # archives could not be merged without duplicates. Every
                # component here comes from inside the recording tree, so the
                # name is a property of the recording.
                rec_name = "_".join((path.name, *dir.relative_to(path).parts))
                if rec_name in fp:
                    # a consequence of the name being reproducible: importing
                    # the same recording twice is now something the file can
                    # detect, rather than quietly producing a second copy under
                    # a name that differed only by where the tree was mounted
                    log.error(
                        "- '%s' is already in %s, skipping", rec_name, args.arffile
                    )
                    continue
                entry = arf.create_entry(
                    fp,
                    name=rec_name,
                    timestamp=timestamp,
                    entry_creator="org.meliza.arfx/arfx-oephys " + core.__version__,
                    **rec.attrs,
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
                                **info,
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
                            **dset.attrs,
                        )


if __name__ == "__main__":
    script()
