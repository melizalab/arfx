# -*- mode: python -*-
"""Tests for arfx.collect (the arfx-collect-sampled script).

collect flattens sampled channels across entries into one 2-D array, which is
how a recording gets handed to a spike sorter. It is the only module that uses
arf.count_channels and one of two that use arf.keys_by_creation.

The consistency checks are tested directly because they return data structures
that are awkward to observe through the CLI; the packing and output format are
tested through collect_sampled_script.
"""

import arf
import numpy as np
import pytest
from conftest import CHANNELS, ENTRY_SAMPLES, SAMPLING_RATE, make_sampled_file

from arfx import collect, io

NENTRIES = 3


@pytest.fixture
def src(sampled_arf_file):
    with arf.open_file(sampled_arf_file, "r") as fp:
        yield fp


def run_collect(src_path, out_path, *extra):
    collect.collect_sampled_script([*extra, str(src_path), str(out_path)])
    return out_path


# ------------------------------------------------------------------- helpers


def test_first_returns_first_subdict_value():
    d = {"a": {"x": 1}, "b": {"x": 2}}
    assert collect.first(d, lambda v: v["x"]) == 1


def test_all_items_equal():
    assert collect.all_items_equal({"a": {"x": 1}, "b": {"x": 1}}, lambda v: v["x"])
    assert not collect.all_items_equal({"a": {"x": 1}, "b": {"x": 2}}, lambda v: v["x"])


def test_all_items_equal_on_empty_dict():
    # `first` returns None for an empty dict, so a caller that got here with no
    # matching channels would go on to use None as a sampling rate
    assert collect.all_items_equal({}, lambda v: v["x"])
    assert collect.first({}, lambda v: v["x"]) is None


# -------------------------------------------------------- channel_properties


def test_channel_properties_reports_required_fields(src):
    props = collect.channel_properties(src["entry_000"])
    assert set(props) == set(CHANNELS)
    for channel in CHANNELS:
        assert props[channel]["sampling_rate"] == SAMPLING_RATE
        assert props[channel]["samples"] == ENTRY_SAMPLES
        assert props[channel]["channels"] == 1
        assert props[channel]["dtype"] == np.dtype("h")


def test_channel_properties_honors_channel_filter(src):
    props = collect.channel_properties(src["entry_000"], channels=["chan_a"])
    assert set(props) == {"chan_a"}


def test_channel_properties_honors_predicate(src):
    props = collect.channel_properties(src["entry_000"], predicate=lambda dset: False)
    assert props == {}


# ---------------------------------------------------- check_entry_consistency


def test_check_entry_consistency_accepts_uniform_file(src):
    entry_names, props = collect.check_entry_consistency(src)
    assert entry_names == [f"entry_{i:03}" for i in range(NENTRIES)]
    assert set(props) == set(CHANNELS)
    # `samples` is popped during the check, since it is allowed to differ
    assert "samples" not in props["chan_a"]


def test_check_entry_consistency_returns_entries_in_creation_order(tmp_path):
    path = tmp_path / "unordered.arf"
    with arf.open_file(path, "w") as fp:
        for name in ("zebra", "alpha", "middle"):
            entry = arf.create_entry(fp, name, 1000)
            arf.create_dataset(
                entry, "chan", np.zeros(100, dtype="h"), sampling_rate=SAMPLING_RATE
            )
    with arf.open_file(path, "r") as fp:
        entry_names, _ = collect.check_entry_consistency(fp)
    assert entry_names == ["zebra", "alpha", "middle"]


def test_check_entry_consistency_rejects_mismatched_sample_counts(tmp_path, caplog):
    path = tmp_path / "ragged.arf"
    with arf.open_file(path, "w") as fp:
        entry = arf.create_entry(fp, "entry", 1000)
        arf.create_dataset(
            entry, "chan_a", np.zeros(100, dtype="h"), sampling_rate=SAMPLING_RATE
        )
        arf.create_dataset(
            entry, "chan_b", np.zeros(200, dtype="h"), sampling_rate=SAMPLING_RATE
        )
    with arf.open_file(path, "r") as fp:
        assert collect.check_entry_consistency(fp) is None
    assert "sample count differs" in caplog.text


def test_check_entry_consistency_rejects_mismatched_channels(tmp_path, caplog):
    path = tmp_path / "mismatch.arf"
    with arf.open_file(path, "w") as fp:
        for i, channel in enumerate(("chan_a", "chan_b")):
            entry = arf.create_entry(fp, f"entry_{i}", 1000 + i)
            arf.create_dataset(
                entry, channel, np.zeros(100, dtype="h"), sampling_rate=SAMPLING_RATE
            )
    with arf.open_file(path, "r") as fp:
        assert collect.check_entry_consistency(fp) is None
    assert "do not match" in caplog.text


def test_check_entry_consistency_entries_filter_is_inverted(src):
    # CHARACTERIZATION: this is a bug. The docstring says "if not None, restrict
    # to entries with supplied names" and -e/--entries is documented as "list of
    # entries to unpack", but the guard is
    #     if entries is not None and entry_name in entries: continue
    # so naming an entry EXCLUDES it. Should be `not in`.
    entry_names, _ = collect.check_entry_consistency(src, entries=["entry_000"])
    assert entry_names == ["entry_001", "entry_002"]


# ---------------------------------------------------------- iter_entry_chunks


def test_iter_entry_chunks_covers_every_sample(src):
    chunks = list(collect.iter_entry_chunks(src["entry_000"], None, arf.is_time_series))
    total = sum(chunk.shape[0] for chunk in chunks)
    assert total == ENTRY_SAMPLES
    assert all(chunk.shape[1] == len(CHANNELS) for chunk in chunks)


def test_iter_entry_chunks_interleaves_channels(src):
    stacked = np.concatenate(
        list(collect.iter_entry_chunks(src["entry_000"], None, arf.is_time_series))
    )
    # conftest writes chan_a as a ramp and chan_b as the same ramp offset by
    # ENTRY_SAMPLES, so the column order is observable
    assert stacked[0, 0] == 0
    assert stacked[0, 1] == ENTRY_SAMPLES
    assert stacked[-1, 0] == ENTRY_SAMPLES - 1


# ---------------------------------------------------------------- script: dat


def test_collect_to_dat(tmp_path, sampled_arf_file):
    # raw dat carries no header, so the reader has to be told the layout that
    # collect wrote -- this is the format's whole point for spike sorters
    out = run_collect(sampled_arf_file, tmp_path / "out.dat")
    with io.open(out, "r", dtype="h", nchannels=len(CHANNELS)) as fp:
        data = fp.read()
    assert data.shape == (NENTRIES * ENTRY_SAMPLES, len(CHANNELS))
    # entries are concatenated in natsorted order, each a fresh ramp
    assert data[0, 0] == 0
    assert data[ENTRY_SAMPLES, 0] == 0
    assert data[0, 1] == ENTRY_SAMPLES


def test_collect_to_wav(tmp_path, sampled_arf_file):
    # wav does carry the layout, so it round-trips without help
    out = run_collect(sampled_arf_file, tmp_path / "out.wav")
    with io.open(out, "r") as fp:
        assert fp.sampling_rate == SAMPLING_RATE
        assert fp.nchannels == len(CHANNELS)
        data = fp.read()
    assert data.shape == (NENTRIES * ENTRY_SAMPLES, len(CHANNELS))


def test_collect_converts_dtype(tmp_path, sampled_arf_file):
    out = run_collect(sampled_arf_file, tmp_path / "out.wav", "-d", "f")
    with io.open(out, "r") as fp:
        assert fp.read().dtype == np.dtype("f")


def test_collect_channel_subset(tmp_path, sampled_arf_file):
    # --channels is nargs="+", so the = form is required to stop it swallowing
    # the positional arguments
    out = run_collect(sampled_arf_file, tmp_path / "out.wav", "--channels=chan_a")
    with io.open(out, "r") as fp:
        assert fp.nchannels == 1


def test_collect_channel_file(tmp_path, sampled_arf_file):
    channel_file = tmp_path / "channels.txt"
    channel_file.write_text("# a comment\nchan_b\n")
    out = run_collect(sampled_arf_file, tmp_path / "out.dat", "-C", str(channel_file))
    with io.open(out, "r") as fp:
        assert fp.nchannels == 1
        # chan_b is the ramp offset by ENTRY_SAMPLES
        assert fp.read()[0] == ENTRY_SAMPLES


def test_collect_dry_run_writes_nothing(tmp_path, sampled_arf_file):
    out = tmp_path / "out.dat"
    collect.collect_sampled_script(["--dry-run", str(sampled_arf_file), str(out)])
    assert not out.exists()


def test_collect_stop_truncates_output(tmp_path, sampled_arf_file):
    # --stop is applied per chunk, so the cutoff is the first chunk boundary at
    # or past the requested sample count rather than the exact sample
    out = run_collect(sampled_arf_file, tmp_path / "out.dat", "--stop", "1000")
    with io.open(out, "r") as fp:
        written = fp.read().shape[0]
    assert 0 < written < NENTRIES * ENTRY_SAMPLES


def test_collect_mountain_params(tmp_path, sampled_arf_file):
    import json

    run_collect(sampled_arf_file, tmp_path / "out.dat", "--mountain-params")
    params = json.loads((tmp_path / "params.json").read_text())
    assert params == {"samplerate": SAMPLING_RATE, "spike_sign": -1}


def test_collect_warns_on_mixed_sampling_rates(tmp_path, caplog):
    path = tmp_path / "mixed.arf"
    with arf.open_file(path, "w") as fp:
        entry = arf.create_entry(fp, "entry", 1000)
        arf.create_dataset(
            entry, "chan_a", np.zeros(100, dtype="h"), sampling_rate=1000
        )
        arf.create_dataset(
            entry, "chan_b", np.zeros(100, dtype="h"), sampling_rate=2000
        )
    run_collect(path, tmp_path / "out.dat")
    assert "same sampling rate" in caplog.text


def test_collect_on_inconsistent_file_raises_typeerror(tmp_path):
    # CHARACTERIZATION: check_entry_consistency returns None on failure, but the
    # caller unpacks the result into two names without checking, so an
    # inconsistent file exits with an unhandled TypeError after having already
    # logged a clear diagnostic. It should exit non-zero with the message.
    path = make_sampled_file(tmp_path / "a.arf", nentries=1)
    with arf.open_file(path, "a") as fp:
        entry = arf.create_entry(fp, "extra", 9999)
        arf.create_dataset(
            entry, "other", np.zeros(100, dtype="h"), sampling_rate=SAMPLING_RATE
        )
    with pytest.raises(TypeError):
        run_collect(path, tmp_path / "out.dat")
