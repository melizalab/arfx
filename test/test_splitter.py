# -*- mode: python -*-
"""Tests for arfx.splitter (the arfx-split script).

splitter rechunks long recordings into fixed-duration entries. The chunking
arithmetic and the jill_log merge are tested directly; the file-level behavior
(ordering across sources, append mode, attribute provenance) goes through
main().
"""

import datetime

import arf
import h5py as h5
import numpy as np
import pytest
from conftest import CHANNELS, SAMPLING_RATE, make_sampled_file, tstamp

from arfx import splitter

# make_sampled_file writes 5000 samples at 1000 Hz, so each entry is 5 seconds
ENTRY_DURATION = 5.0

LOG_DTYPE = [("sec", "i8"), ("usec", "i8"), ("message", h5.string_dtype())]


def add_jill_log(path, records):
    with h5.File(path, "a") as fp:
        fp.create_dataset("jill_log", data=np.array(records, dtype=np.dtype(LOG_DTYPE)))
    return path


def run_split(tgt, *srcs, **kwargs):
    argv = []
    for key, value in kwargs.items():
        argv.extend([f"--{key}", str(value)])
    splitter.main([*argv, *(str(s) for s in srcs), str(tgt)])
    return tgt


# --------------------------------------------------------------- entry helpers


def one_entry_run(entry):
    """The run a single unspliced entry forms, as find_runs would build it"""
    return [splitter.Piece(entry, tstamp, 0)]


def run_spans(run):
    entry = run[0].entry
    spans = {
        name: splitter.run_pieces(run, name)
        for name in splitter.sampled_datasets(entry)
    }
    totals = {
        name: sum(dset.shape[0] - skip for dset, skip in pieces)
        for name, pieces in spans.items()
    }
    return spans, totals


def test_run_duration_uses_longest_sampled_dataset(tmp_path):
    path = tmp_path / "d.arf"
    with arf.open_file(path, "w") as fp:
        entry = arf.create_entry(fp, "entry", tstamp)
        arf.create_dataset(
            entry, "short", np.zeros(1000, dtype="h"), sampling_rate=SAMPLING_RATE
        )
        arf.create_dataset(
            entry, "long", np.zeros(4000, dtype="h"), sampling_rate=SAMPLING_RATE
        )
        # event data must not count toward the duration
        arf.create_dataset(entry, "spikes", np.arange(99999.0), units="s")
        spans, totals = run_spans(one_entry_run(entry))
        assert splitter.run_duration(spans, totals) == 4.0


def test_run_duration_of_entry_with_no_sampled_data(tmp_path):
    path = tmp_path / "d.arf"
    with arf.open_file(path, "w") as fp:
        entry = arf.create_entry(fp, "entry", tstamp)
        arf.create_dataset(entry, "spikes", np.arange(10.0), units="s")
        spans, totals = run_spans(one_entry_run(entry))
        assert splitter.run_duration(spans, totals) == 0
        # and it still gets a chunk, so its attributes survive the split
        assert splitter.run_chunk_count(totals, {}) == 1


def test_entry_timestamps_skips_non_groups(tmp_path):
    path = make_sampled_file(tmp_path / "t.arf", nentries=2)
    with h5.File(path, "a") as fp:
        fp.create_dataset("toplevel", data=np.arange(10))
    with h5.File(path, "r") as fp:
        pairs = list(splitter.entry_timestamps(fp))
    assert len(pairs) == 2
    assert all(isinstance(ts, datetime.datetime) for _, ts in pairs)


# ------------------------------------------------------------------ jill logs


def test_pad_log_messages_fixes_width():
    arr = np.array(
        [(1, 0, b"short"), (2, 0, b"a much longer message")], dtype=np.dtype(LOG_DTYPE)
    )
    padded = splitter.pad_log_messages(arr)
    assert padded.dtype["message"].itemsize == len(b"a much longer message")
    assert padded["message"].tolist() == [b"short", b"a much longer message"]


def test_pad_log_messages_requires_message_field():
    arr = np.array([(1, 0)], dtype=[("sec", "i8"), ("usec", "i8")])
    with pytest.raises(ValueError, match="'message' field"):
        splitter.pad_log_messages(arr)


def test_merge_jill_logs_sorts_by_time(tmp_path):
    a = add_jill_log(
        make_sampled_file(tmp_path / "a.arf", nentries=1), [(2, 0, b"second")]
    )
    b = add_jill_log(
        make_sampled_file(tmp_path / "b.arf", nentries=1),
        [(1, 500, b"first"), (3, 0, b"third")],
    )
    with h5.File(a, "r") as fa, h5.File(b, "r") as fb:
        merged = splitter.merge_jill_logs([fa, fb])
    assert merged["message"].tolist() == [b"first", b"second", b"third"]


def test_merge_jill_logs_returns_none_when_absent(tmp_path):
    path = make_sampled_file(tmp_path / "a.arf", nentries=1)
    with h5.File(path, "r") as fp:
        assert splitter.merge_jill_logs([fp]) is None


def test_jill_log_is_merged_into_output(tmp_path):
    src = add_jill_log(
        make_sampled_file(tmp_path / "a.arf", nentries=1), [(1, 0, b"hello")]
    )
    tgt = run_split(tmp_path / "out.arf", src, duration=ENTRY_DURATION)
    with arf.open_file(tgt, "r") as fp:
        assert fp["jill_log"]["message"].tolist() == [b"hello"]


def test_jill_log_is_not_merged_when_appending(tmp_path):
    src = add_jill_log(
        make_sampled_file(tmp_path / "a.arf", nentries=1), [(1, 0, b"hello")]
    )
    tgt = tmp_path / "out.arf"
    splitter.main(["--append", "--duration", "5.0", str(src), str(tgt)])
    with arf.open_file(tgt, "r") as fp:
        assert "jill_log" not in fp


# -------------------------------------------------------------------- chunking


def test_split_into_even_chunks(tmp_path):
    src = make_sampled_file(tmp_path / "a.arf", nentries=1)
    tgt = run_split(tmp_path / "out.arf", src, duration=2.0)
    with arf.open_file(tgt, "r") as fp:
        names = sorted(n for n in fp if arf.is_entry(fp[n]))
        assert names == ["entry_00000", "entry_00001", "entry_00002"]
        # 5 seconds at 2 s per chunk: 2000 + 2000 + 1000
        assert [fp[n]["chan_a"].shape[0] for n in names] == [2000, 2000, 1000]


def test_exactly_divisible_duration_leaves_no_empty_entry(tmp_path):
    # The chunk count used to be `int(max_duration // duration) + 1`, which
    # added a zero-sample entry whenever the length divided evenly. It is now
    # counted per dataset in samples, so the +1 only appears when there really
    # is a partial chunk to hold.
    src = make_sampled_file(tmp_path / "a.arf", nentries=1)
    tgt = run_split(tmp_path / "out.arf", src, duration=ENTRY_DURATION)
    with arf.open_file(tgt, "r") as fp:
        names = sorted(n for n in fp if arf.is_entry(fp[n]))
        assert len(names) == 1
        assert fp[names[0]]["chan_a"].shape[0] == 5000


def test_partial_final_chunk_is_kept(tmp_path):
    # the complement: a length that does not divide evenly still gets its
    # remainder written, rather than being truncated
    src = make_sampled_file(tmp_path / "a.arf", nentries=1)
    tgt = run_split(tmp_path / "out.arf", src, duration=2.0)
    with arf.open_file(tgt, "r") as fp:
        names = sorted(n for n in fp if arf.is_entry(fp[n]))
        assert [fp[n]["chan_a"].shape[0] for n in names] == [2000, 2000, 1000]


def test_entry_without_sampled_data_is_preserved(tmp_path):
    # an entry holding only event data has no duration, but its attributes and
    # datasets should survive rather than being dropped for want of a chunk
    path = tmp_path / "events.arf"
    with arf.open_file(path, "w") as fp:
        entry = arf.create_entry(fp, "entry", tstamp, experimenter="dmeliza")
        arf.create_dataset(entry, "spikes", np.arange(10.0), units="s")
    tgt = run_split(tmp_path / "out.arf", path, duration=2.0)
    with arf.open_file(tgt, "r") as fp:
        names = [n for n in fp if arf.is_entry(fp[n])]
        assert len(names) == 1
        assert fp[names[0]].attrs["experimenter"] == "dmeliza"


def test_duration_shorter_than_one_sample_is_rejected(tmp_path):
    src = make_sampled_file(tmp_path / "a.arf", nentries=1)
    with pytest.raises(ValueError, match="less than one sample"):
        run_split(tmp_path / "out.arf", src, duration=1e-9)


def test_chunk_timestamps_advance_by_duration(tmp_path):
    src = make_sampled_file(tmp_path / "a.arf", nentries=1)
    tgt = run_split(tmp_path / "out.arf", src, duration=2.0)
    with arf.open_file(tgt, "r") as fp:
        times = [
            arf.timestamp_to_float(fp[n].attrs["timestamp"])
            for n in sorted(n for n in fp if arf.is_entry(fp[n]))
        ]
    assert times[1] - times[0] == pytest.approx(2.0)
    assert times[2] - times[1] == pytest.approx(2.0)


def test_chunk_data_is_contiguous(tmp_path):
    # conftest writes chan_a as a ramp, so the chunks must join without a gap
    src = make_sampled_file(tmp_path / "a.arf", nentries=1)
    tgt = run_split(tmp_path / "out.arf", src, duration=2.0)
    with arf.open_file(tgt, "r") as fp:
        joined = np.concatenate(
            [fp[n]["chan_a"][:] for n in sorted(n for n in fp if arf.is_entry(fp[n]))]
        )
    assert joined.tolist() == list(range(5000))


def test_all_channels_are_split(tmp_path):
    src = make_sampled_file(tmp_path / "a.arf", nentries=1)
    tgt = run_split(tmp_path / "out.arf", src, duration=2.0)
    with arf.open_file(tgt, "r") as fp:
        assert set(fp["entry_00000"].keys()) == set(CHANNELS)


def test_event_datasets_are_skipped(tmp_path):
    path = tmp_path / "mixed.arf"
    with arf.open_file(path, "w") as fp:
        entry = arf.create_entry(fp, "entry", tstamp)
        arf.create_dataset(
            entry, "chan", np.arange(2000, dtype="h"), sampling_rate=SAMPLING_RATE
        )
        arf.create_dataset(entry, "spikes", np.arange(10.0), units="s")
    tgt = run_split(tmp_path / "out.arf", path, duration=1.0)
    with arf.open_file(tgt, "r") as fp:
        assert "spikes" not in fp["entry_00000"]
        assert "chan" in fp["entry_00000"]


# ------------------------------------------------------------------ provenance


def test_origin_attributes_are_recorded(tmp_path):
    src = make_sampled_file(tmp_path / "a.arf", nentries=1, experimenter="dmeliza")
    tgt = run_split(tmp_path / "out.arf", src, duration=2.0)
    with arf.open_file(tgt, "r") as fp:
        attrs = fp["entry_00000"].attrs
        assert attrs["origin-file"] == "a.arf"
        assert attrs["origin-entry"] == "entry_000"
        assert attrs["experimenter"] == "dmeliza"
        # the source uuid is preserved under a different name, and the target
        # gets a fresh one of its own
        assert "origin-uuid" in attrs
        assert attrs["uuid"] != attrs["origin-uuid"]


def test_dataset_uuid_is_renamed(tmp_path):
    path = tmp_path / "u.arf"
    with arf.open_file(path, "w") as fp:
        entry = arf.create_entry(fp, "entry", tstamp)
        dset = arf.create_dataset(
            entry, "chan", np.zeros(1000, dtype="h"), sampling_rate=SAMPLING_RATE
        )
        arf.set_uuid(dset, "0" * 32)
    tgt = run_split(tmp_path / "out.arf", path, duration=1.0)
    with arf.open_file(tgt, "r") as fp:
        attrs = fp["entry_00000"]["chan"].attrs
        origin = attrs["origin-uuid"]
        if isinstance(origin, bytes):
            origin = origin.decode()
        assert origin.replace("-", "") == "0" * 32
        assert "uuid" not in attrs


# ------------------------------------------------------------- multiple inputs


def test_entries_are_ordered_by_timestamp_across_files(tmp_path):
    # b starts an hour before a, so the output must lead with b's entry
    a = make_sampled_file(tmp_path / "a.arf", nentries=1, start=tstamp)
    b = make_sampled_file(tmp_path / "b.arf", nentries=1, start=tstamp - 3600)
    tgt = run_split(tmp_path / "out.arf", a, b, duration=ENTRY_DURATION)
    with arf.open_file(tgt, "r") as fp:
        origins = [
            fp[n].attrs["origin-file"]
            for n in sorted(n for n in fp if arf.is_entry(fp[n]))
        ]
    assert origins[0] == "b.arf"
    assert "a.arf" in origins


def test_append_continues_entry_numbering(tmp_path):
    a = make_sampled_file(tmp_path / "a.arf", nentries=1)
    b = make_sampled_file(tmp_path / "b.arf", nentries=1)
    tgt = tmp_path / "out.arf"
    run_split(tgt, a, duration=ENTRY_DURATION)
    splitter.main(["--append", "--duration", str(ENTRY_DURATION), str(b), str(tgt)])
    with arf.open_file(tgt, "r") as fp:
        names = sorted(n for n in fp if arf.is_entry(fp[n]))
        # one chunk per source, numbered on from where the first run stopped
        assert names == [f"entry_{i:05}" for i in range(2)]


# --------------------------------------------------------------------- dry run


def test_dry_run_writes_nothing(tmp_path):
    src = make_sampled_file(tmp_path / "a.arf", nentries=1)
    tgt = tmp_path / "out.arf"
    splitter.main(["--dry-run", "--duration", "2.0", str(src), str(tgt)])
    assert not tgt.exists()


# ------------------------------------------------------------------ splicing


def make_contiguous_file(path, *, lengths, first_frame=0, gaps=None, rate=1000):
    """A file whose entries abut on the sample timeline, as a recorder writes them.

    `gaps` gives the signed sample distance between consecutive entries: 0 for
    contiguous (the default), negative for an overlap, positive for a real gap.
    The data are one continuous ramp across the whole recording, so a spliced
    result can be checked against `numpy.arange` and any misplaced sample shows
    up as a discontinuity.
    """
    gaps = gaps or [0] * (len(lengths) - 1)
    frame = first_frame
    position = 0
    with arf.open_file(path, "w") as fp:
        for i, length in enumerate(lengths):
            entry = arf.create_entry(
                fp,
                f"e{i:03}",
                tstamp + position / rate,
                jack_frame=np.uint32(frame % (2**32)),
                jack_sampling_rate=np.uint32(rate),
            )
            arf.create_dataset(
                entry,
                "pcm",
                np.arange(position, position + length, dtype="i4"),
                sampling_rate=rate,
            )
            if i < len(lengths) - 1:
                frame = (frame + length + gaps[i]) % (2**32)
                position += length + gaps[i]
    return path


def spliced(tgt):
    with arf.open_file(tgt, "r") as fp:
        names = [n for n in arf.keys_by_creation(fp) if arf.is_entry(fp[n])]
        return [fp[n]["pcm"][:] for n in names]


def test_contiguous_entries_are_spliced_into_full_chunks(tmp_path):
    # three 1000-sample entries abutting exactly, chunked at 1.5 s = 1500
    # samples. Without splicing each entry chunks separately and the output is
    # 1000/1000/1000; spliced it is one 3000-sample stream cut at 1500.
    src = make_contiguous_file(tmp_path / "a.arf", lengths=[1000, 1000, 1000])
    chunks = spliced(run_split(tmp_path / "t.arf", src, duration=1.5))
    assert [len(c) for c in chunks] == [1500, 1500]
    assert np.array_equal(np.concatenate(chunks), np.arange(3000))


def test_no_splice_restores_per_entry_chunking(tmp_path):
    src = make_contiguous_file(tmp_path / "a.arf", lengths=[1000, 1000, 1000])
    tgt = tmp_path / "t.arf"
    splitter.main(["--no-splice", "--duration", "1.5", str(src), str(tgt)])
    assert [len(c) for c in spliced(tgt)] == [1000, 1000, 1000]


def test_a_gap_breaks_the_run(tmp_path):
    # the second entry starts 500 samples after the first ends, so the two are
    # separate recordings and must not be joined
    src = make_contiguous_file(tmp_path / "a.arf", lengths=[1000, 1000], gaps=[500])
    chunks = spliced(run_split(tmp_path / "t.arf", src, duration=1.5))
    assert [len(c) for c in chunks] == [1000, 1000]


def test_overlapping_entries_are_spliced_without_duplicating_samples(tmp_path):
    # the second entry repeats the last 200 samples of the first
    src = make_contiguous_file(tmp_path / "a.arf", lengths=[1000, 1000], gaps=[-200])
    chunks = spliced(run_split(tmp_path / "t.arf", src, duration=10))
    assert len(chunks) == 1
    # 1000 + 1000 - 200 duplicated, and still a single unbroken ramp
    assert np.array_equal(chunks[0], np.arange(1800))


def test_overlap_beyond_the_limit_is_not_spliced(tmp_path):
    src = make_contiguous_file(tmp_path / "a.arf", lengths=[1000, 1000], gaps=[-200])
    tgt = tmp_path / "t.arf"
    splitter.main(["--max-overlap", "100", "--duration", "10", str(src), str(tgt)])
    assert [len(c) for c in spliced(tgt)] == [1000, 1000]


def test_overlap_with_mismatched_data_is_refused(tmp_path, caplog):
    # the frame counter claims an overlap but the samples there disagree, so
    # the counter and the data cannot both be right
    src = make_contiguous_file(tmp_path / "a.arf", lengths=[1000, 1000], gaps=[-200])
    with arf.open_file(src, "r+") as fp:
        fp["e001"]["pcm"][:200] = -1
    chunks = spliced(run_split(tmp_path / "t.arf", src, duration=10))
    assert "data there differ" in caplog.text
    assert [len(c) for c in chunks] == [1000, 1000]


def test_splicing_across_the_frame_counter_wrap(tmp_path):
    # jack_frame is a uint32, and at 44.1 kHz it wraps about every 27 hours --
    # which is why a long recording gets broken into entries at all. An entry
    # starting just after the wrap must still read as contiguous.
    src = make_contiguous_file(
        tmp_path / "a.arf", lengths=[1000, 1000], first_frame=2**32 - 500
    )
    with arf.open_file(src, "r") as fp:
        # the second entry really did wrap round
        assert fp["e001"].attrs["jack_frame"] < fp["e000"].attrs["jack_frame"]
    chunks = spliced(run_split(tmp_path / "t.arf", src, duration=10))
    assert len(chunks) == 1
    assert np.array_equal(chunks[0], np.arange(2000))


def test_entries_are_spliced_across_source_files(tmp_path):
    # the recording continues in a second file, which is the case the issue
    # names: entries are sorted by timestamp before the run is assembled
    a = make_contiguous_file(tmp_path / "a.arf", lengths=[1000])
    b = make_contiguous_file(tmp_path / "b.arf", lengths=[1000], first_frame=1000)
    with arf.open_file(b, "r+") as fp:
        fp["e000"].attrs["timestamp"] = arf.convert_timestamp(tstamp + 1.0)
        fp["e000"]["pcm"][:] = np.arange(1000, 2000)
    chunks = spliced(run_split(tmp_path / "t.arf", a, b, duration=10))
    assert len(chunks) == 1
    assert np.array_equal(chunks[0], np.arange(2000))


def test_entries_without_a_frame_attribute_are_never_spliced(tmp_path):
    # make_sampled_file writes no jack_frame, so there is nothing to decide
    # contiguity from and every entry stays its own run
    src = make_sampled_file(tmp_path / "a.arf", nentries=2)
    tgt = run_split(tmp_path / "t.arf", src, duration=10)
    with arf.open_file(tgt, "r") as fp:
        entries = [n for n in arf.keys_by_creation(fp) if arf.is_entry(fp[n])]
        assert len(entries) == 2
        assert all(fp[n][CHANNELS[0]].shape[0] == 5000 for n in entries)


def test_frame_attribute_advances_with_the_chunk(tmp_path):
    # every chunk used to inherit the source entry's frame counter, so they all
    # claimed the same start and this script's own output could not be spliced
    src = make_contiguous_file(tmp_path / "a.arf", lengths=[3000], first_frame=1000)
    tgt = run_split(tmp_path / "t.arf", src, duration=1.0)
    with arf.open_file(tgt, "r") as fp:
        frames = [
            int(fp[n].attrs["jack_frame"])
            for n in arf.keys_by_creation(fp)
            if arf.is_entry(fp[n])
        ]
    assert frames == [1000, 2000, 3000]


def test_split_output_can_be_spliced_again(tmp_path):
    # the round trip the issue asks for: split a recording, then re-chunk the
    # result at a different duration and get the same samples back
    src = make_contiguous_file(tmp_path / "a.arf", lengths=[6000], first_frame=7)
    once = run_split(tmp_path / "once.arf", src, duration=1.0)
    assert len(spliced(once)) == 6
    twice = run_split(tmp_path / "twice.arf", once, duration=3.0)
    chunks = spliced(twice)
    assert [len(c) for c in chunks] == [3000, 3000]
    assert np.array_equal(np.concatenate(chunks), np.arange(6000))


def test_origin_entry_names_the_entry_a_chunk_starts_in(tmp_path):
    src = make_contiguous_file(tmp_path / "a.arf", lengths=[1000, 1000])
    tgt = run_split(tmp_path / "t.arf", src, duration=1.5)
    with arf.open_file(tgt, "r") as fp:
        origins = [
            fp[n].attrs["origin-entry"]
            for n in arf.keys_by_creation(fp)
            if arf.is_entry(fp[n])
        ]
    # [0:1500) starts in e000; [1500:2000) starts in e001
    assert origins == ["e000", "e001"]
