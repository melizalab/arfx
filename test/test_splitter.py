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


def test_entry_duration_uses_longest_sampled_dataset(tmp_path):
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
        assert splitter.entry_duration(entry) == 4.0


def test_entry_duration_of_entry_with_no_sampled_data(tmp_path):
    path = tmp_path / "d.arf"
    with arf.open_file(path, "w") as fp:
        entry = arf.create_entry(fp, "entry", tstamp)
        arf.create_dataset(entry, "spikes", np.arange(10.0), units="s")
        assert splitter.entry_duration(entry) == 0


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


def test_exactly_divisible_duration_leaves_an_empty_trailing_entry(tmp_path):
    # CHARACTERIZATION: n_chunks is `int(max_duration // duration) + 1`, so a
    # recording whose length is an exact multiple of the chunk duration gets one
    # extra entry containing zero samples. The +1 is there to catch the partial
    # final chunk and does not check whether one exists.
    src = make_sampled_file(tmp_path / "a.arf", nentries=1)
    tgt = run_split(tmp_path / "out.arf", src, duration=ENTRY_DURATION)
    with arf.open_file(tgt, "r") as fp:
        names = sorted(n for n in fp if arf.is_entry(fp[n]))
        assert len(names) == 2
        assert fp[names[0]]["chan_a"].shape[0] == 5000
        assert fp[names[1]]["chan_a"].shape[0] == 0


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
        # two chunks per source (the second is the empty trailing one)
        assert names == [f"entry_{i:05}" for i in range(4)]


# --------------------------------------------------------------------- dry run


def test_dry_run_writes_nothing(tmp_path):
    src = make_sampled_file(tmp_path / "a.arf", nentries=1)
    tgt = tmp_path / "out.arf"
    splitter.main(["--dry-run", "--duration", "2.0", str(src), str(tgt)])
    assert not tgt.exists()
