# -*- mode: python -*-
"""This script collects data from a recording, possibly made over multiple ARF
files and splits it into chunks in a new file. Each new entry is
given an updated timestamp and attributes from the source entries.

Entries that are contiguous on the sample timeline are spliced back together
before chunking, so a recording that was broken up by the recorder or by an
earlier run of this script is re-chunked as one continuous stream. Only sampled
datasets are processed.

"""

import argparse
import datetime
import itertools
import logging
import operator
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import NamedTuple

import arf
import h5py as h5
import numpy as np

log = logging.getLogger("arfx-split")  # root logger

# jrecord stores the recorder's frame counter here. It is not part of the ARF
# specification, which is why --frame-attr exists.
DEFAULT_FRAME_ATTR = "jack_frame"
# that counter is a uint32, and at 44.1 kHz it wraps about every 27 hours --
# well inside the range of recording this script is meant for, and one of the
# reasons a recording gets broken into several entries in the first place
FRAME_MODULUS = 2**32


def entry_timestamps(
    arf_file: h5.Group,
) -> Iterator[tuple[h5.Group, datetime.datetime]]:
    """Iterate through entries in arf file, yielding a seq of (entry, timestamp) tuples"""
    for _entry_name, entry in arf_file.items():
        if not isinstance(entry, h5.Group):
            continue
        entry_time = arf.timestamp_to_datetime(entry.attrs["timestamp"])
        yield (entry, entry_time)


def run_duration(spans: dict, totals: dict[str, int]) -> float:
    """Duration in seconds of the longest sampled stream in a run"""
    return max(
        (
            total / float(spans[name][0][0].attrs["sampling_rate"])
            for name, total in totals.items()
        ),
        default=0.0,
    )


def run_chunk_sizes(spans: dict, duration: float) -> dict[str, int]:
    """Chunk length in samples for each stream in a run"""
    sizes = {}
    for name, pieces in spans.items():
        sampling_rate = pieces[0][0].attrs["sampling_rate"]
        chunk_size = int(duration * sampling_rate)
        if chunk_size < 1:
            raise ValueError(
                f"a duration of {duration} s is less than one sample at "
                f"{sampling_rate} Hz"
            )
        sizes[name] = chunk_size
    return sizes


def run_chunk_count(totals: dict[str, int], chunk_sizes: dict[str, int]) -> int:
    """Number of chunks needed to cover a run.

    Counted per stream in samples rather than from the duration in seconds, so
    that floating-point error cannot tack on a spurious chunk. A run with no
    sampled data still yields one chunk, so that its attributes and any event
    datasets survive the split.
    """
    return (
        max(
            (-(-total // chunk_sizes[name]) for name, total in totals.items()),
            default=1,
        )
        or 1
    )


def sampled_datasets(entry: h5.Group) -> dict[str, h5.Dataset]:
    """The sampled datasets of an entry, keyed by name"""
    return {name: dset for name, dset in entry.items() if arf.is_time_series(dset)}


def entry_sample_count(entry: h5.Group) -> int | None:
    """How many samples the entry spans, or None if that is not well defined.

    None means the entry has no position on a sample timeline and so can never
    be spliced: either it holds no sampled data, or its sampled datasets
    disagree about length or sampling rate, in which case there is no single
    number of samples the next entry would have to start after.
    """
    dsets = sampled_datasets(entry)
    if not dsets:
        return None
    counts = {dset.shape[0] for dset in dsets.values()}
    rates = {float(dset.attrs["sampling_rate"]) for dset in dsets.values()}
    if len(counts) > 1 or len(rates) > 1:
        return None
    return counts.pop()


def entry_frame(entry: h5.Group, attr: str) -> int | None:
    """The recorder's frame counter at the start of the entry, or None"""
    try:
        return int(entry.attrs[attr]) % FRAME_MODULUS
    except (KeyError, TypeError, ValueError):
        return None


def frame_distance(previous_end: int, start: int) -> int:
    """Samples from the end of one entry to the start of the next.

    Zero when they abut exactly, negative when they overlap, positive when
    there is a gap. Both arguments are positions on a counter that wraps, so
    the difference is taken modulo the counter and folded into the half-open
    range around zero -- otherwise an entry recorded just after the counter
    wrapped looks like it starts four billion samples before its predecessor.
    """
    half = FRAME_MODULUS // 2
    return (start - previous_end + half) % FRAME_MODULUS - half


def overlap_is_consistent(previous: h5.Group, entry: h5.Group, overlap: int) -> bool:
    """Check that overlapping entries agree about the samples they share.

    Two entries claiming the same stretch of the timeline should hold the same
    data there. If they do not, the frame counter and the data disagree, and
    splicing them would silently produce a recording that never happened.
    """
    previous_dsets = sampled_datasets(previous)
    for name, dset in sampled_datasets(entry).items():
        before = previous_dsets.get(name)
        if before is None:
            return False
        if overlap > before.shape[0] or overlap > dset.shape[0]:
            return False
        if not np.array_equal(before[-overlap:], dset[:overlap]):
            return False
    return True


class Piece(NamedTuple):
    """One source entry's contribution to a spliced run.

    `skip` is the number of leading samples to drop, which is how an overlap
    with the previous entry is resolved: the samples are already in the stream.
    """

    entry: h5.Group
    timestamp: datetime.datetime
    skip: int


def find_runs(
    entries: Sequence[tuple[h5.Group, datetime.datetime]],
    *,
    frame_attr: str = DEFAULT_FRAME_ATTR,
    max_overlap: int = 0,
    splice: bool = True,
) -> list[list[Piece]]:
    """Group timestamp-ordered entries into runs that are contiguous in samples.

    Each run is chunked as a single continuous recording. An entry starts a new
    run whenever it cannot be shown to continue the previous one, so with
    splicing off -- or on input that carries no frame counter -- every entry is
    its own run and the result is the same as not splicing at all.
    """
    runs: list[list[Piece]] = []
    current: list[Piece] = []
    previous_end: int | None = None

    for entry, timestamp in entries:
        start = entry_frame(entry, frame_attr) if splice else None
        count = entry_sample_count(entry) if splice else None
        skip = None
        if current and start is not None and previous_end is not None:
            distance = frame_distance(previous_end, start)
            if distance == 0:
                skip = 0
            elif -max_overlap <= distance < 0:
                overlap = -distance
                if overlap_is_consistent(current[-1].entry, entry, overlap):
                    log.debug("    overlaps previous entry by %d samples", overlap)
                    skip = overlap
                else:
                    log.warning(
                        "  %s: overlaps the previous entry by %d samples but the "
                        "data there differ; not splicing",
                        entry.name,
                        overlap,
                    )
            elif distance < 0:
                log.warning(
                    "  %s: overlaps the previous entry by %d samples, more than "
                    "--max-overlap (%d); not splicing",
                    entry.name,
                    -distance,
                    max_overlap,
                )

        if skip is None:
            if current:
                runs.append(current)
            current = []
        else:
            log.debug("  %s: continues the previous entry", entry.name)
        current.append(Piece(entry, timestamp, skip or 0))
        previous_end = None if start is None or count is None else start + count

    if current:
        runs.append(current)
    return runs


def run_pieces(run: Sequence[Piece], name: str) -> list[tuple[h5.Dataset, int]]:
    """(dataset, skip) for every entry in the run that carries `name`"""
    out = []
    for piece in run:
        dset = piece.entry.get(name)
        if dset is not None and arf.is_time_series(dset):
            out.append((dset, piece.skip))
    return out


def span_source(
    pieces: Sequence[tuple[h5.Dataset, int]], start: int, stop: int
) -> h5.Group | None:
    """The entry holding the first sample of logical [start, stop)"""
    position = 0
    for dset, skip in pieces:
        length = dset.shape[0] - skip
        if position + length > start and position < stop:
            return dset.parent
        position += length
    return None


def read_span(pieces: Sequence[tuple[h5.Dataset, int]], start: int, stop: int):
    """Read logical samples [start, stop) from a spliced sequence of datasets"""
    out = []
    position = 0
    for dset, skip in pieces:
        length = dset.shape[0] - skip
        if position + length > start and position < stop:
            first = max(start - position, 0)
            last = min(stop - position, length)
            out.append(dset[skip + first : skip + last])
        position += length
        if position >= stop:
            break
    if len(out) == 1:
        return out[0]
    return np.concatenate(out) if out else np.empty(0)


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
        help="the maximum duration of entries (default: %(default).2f seconds)",
        type=float,
        default=600,
    )
    p.add_argument(
        "--compress",
        "-z",
        help="set compression level in output file (default: %(default)d)",
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
        "--no-splice",
        dest="splice",
        action="store_false",
        help="chunk each source entry separately, instead of first splicing "
        "entries that are contiguous on the sample timeline",
    )
    p.add_argument(
        "--frame-attr",
        default=DEFAULT_FRAME_ATTR,
        metavar="NAME",
        help="entry attribute holding the recorder's frame counter, used to "
        "tell whether two entries are contiguous (default: %(default)s). "
        "Entries without it are never spliced.",
    )
    p.add_argument(
        "--max-overlap",
        type=int,
        default=4096,
        metavar="SAMPLES",
        help="splice entries that overlap by up to this many samples, dropping "
        "the duplicates (default: %(default)d). The overlapping samples are "
        "compared first, and entries that disagree there are not spliced.",
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

    # group entries that are contiguous in samples, then chunk each group as
    # though it were one recording
    runs = find_runs(
        entries,
        frame_attr=args.frame_attr,
        max_overlap=args.max_overlap,
        splice=args.splice,
    )
    for run in runs:
        first = run[0]
        entry, timestamp = first.entry, first.timestamp
        if len(run) > 1:
            log.info(
                "source run: %s%s + %d contiguous entr%s",
                Path(entry.file.filename).name,
                entry.name,
                len(run) - 1,
                "y" if len(run) == 2 else "ies",
            )
        else:
            log.info("source entry: %s%s", Path(entry.file.filename).name, entry.name)

        # the spliced length of each sampled dataset, and how many chunks that
        # needs. Names come from the first entry: find_runs only extends a run
        # when the overlap check found the same datasets on both sides.
        spans = {name: run_pieces(run, name) for name in sampled_datasets(entry)}
        totals = {
            name: sum(dset.shape[0] - skip for dset, skip in pieces)
            for name, pieces in spans.items()
        }
        chunk_sizes = run_chunk_sizes(spans, args.duration)
        n_chunks = run_chunk_count(totals, chunk_sizes)
        log.debug(
            "  duration: %3.2f s (chunks=%d)", run_duration(spans, totals), n_chunks
        )

        run_frame = entry_frame(entry, args.frame_attr)
        for i in range(n_chunks):
            tgt_entry_name = f"entry_{tgt_entry_index:05}"
            tgt_timestamp = timestamp + datetime.timedelta(seconds=args.duration) * i
            log.info("  target entry: %s (time=%s)", tgt_entry_name, tgt_timestamp)
            tgt_entry_index += 1
            if not args.dry_run:
                tgt_entry = arf.create_entry(tgt_file, tgt_entry_name, tgt_timestamp)
                for k, v in entry.attrs.items():
                    if k == "timestamp":
                        continue
                    elif k == "uuid":
                        k = "origin-uuid"
                    tgt_entry.attrs[k] = v
                # the frame counter has to advance with the chunk. Copying the
                # source entry's value gave every chunk the same start frame,
                # which made this script's own output impossible to splice.
                if run_frame is not None and chunk_sizes:
                    name = next(iter(chunk_sizes))
                    advanced = (run_frame + i * chunk_sizes[name]) % FRAME_MODULUS
                    tgt_entry.attrs[args.frame_attr] = np.asarray(
                        entry.attrs[args.frame_attr]
                    ).dtype.type(advanced)
                # name the source entry this chunk actually starts in, which
                # for unspliced input is the only one it can come from
                source = entry
                if spans:
                    name = next(iter(spans))
                    start = chunk_sizes[name] * i
                    source = (
                        span_source(spans[name], start, start + chunk_sizes[name])
                        or entry
                    )
                tgt_entry.attrs["origin-file"] = Path(source.file.filename).name
                tgt_entry.attrs["origin-entry"] = Path(source.name).name
            for dset_name, pieces in spans.items():
                chunk_size = chunk_sizes[dset_name]
                start = chunk_size * i
                stop = min(start + chunk_size, totals[dset_name])
                log.debug("    %s: [%d:%d]", dset_name, start, stop)
                if not args.dry_run:
                    tgt_attrs = dict(pieces[0][0].attrs)
                    try:
                        tgt_attrs["origin-uuid"] = tgt_attrs.pop("uuid")
                    except KeyError:
                        pass
                    arf.create_dataset(
                        tgt_entry,
                        dset_name,
                        read_span(pieces, start, stop),
                        compression=args.compress,
                        **tgt_attrs,
                    )
