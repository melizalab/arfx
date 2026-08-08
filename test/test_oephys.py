# -*- mode: python -*-
"""Tests for arfx.oephys (the arfx-oephys script).

open-ephys stores a recording as a directory tree whose layout changed at GUI
version 0.6.0, and oephys.py dispatches on the "GUI version" string in
structure.oebin to handle both. That dispatch is the module's main complexity,
so the fixture is parametrized over both generations and every test runs twice.

The tree builder lives in conftest.py, because the pipeline tests need it too.
Its metadata is copied from real GUI 0.5.3.1 and 1.0.2 recordings rather than
invented; see the note there.
"""

import json

import arf
import numpy as np
import pytest
from conftest import (
    EXPECTED_TIME,
    NCHANNELS,
    NEW,
    NSAMPLES,
    OLD,
    SAMPLE_OFFSET,
    build_tree,
    message_dset,
    only_entry,
    rate,
    recording_dir,
    run_oephys,
    ttl_dset,
)
from packaging.version import Version

from arfx import oephys

# ------------------------------------------------------------------- happy path


def test_creates_one_entry_per_recording(tmp_path, tree):
    tgt = run_oephys(tmp_path / "out.arf", tree)
    with arf.open_file(tgt, "r") as fp:
        entry = only_entry(fp)
        assert arf.timestamp_to_datetime(entry.attrs["timestamp"]) == EXPECTED_TIME


def test_continuous_channels_are_deinterleaved(tmp_path, tree):
    tgt = run_oephys(tmp_path / "out.arf", tree)
    with arf.open_file(tgt, "r") as fp:
        entry = only_entry(fp)
        for j in range(NCHANNELS):
            dset = entry[f"CH{j + 1}"]
            assert dset.shape == (NSAMPLES,)
            assert dset[0] == j * NSAMPLES
            assert dset[-1] == j * NSAMPLES + NSAMPLES - 1


def test_continuous_attributes(tmp_path, tree, gui_version):
    tgt = run_oephys(tmp_path / "out.arf", tree)
    with arf.open_file(tgt, "r") as fp:
        attrs = only_entry(fp)["CH1"].attrs
        assert attrs["sampling_rate"] == rate(gui_version)
        # the offset is the first sample index converted to seconds
        assert attrs["offset"] == pytest.approx(SAMPLE_OFFSET / rate(gui_version))
        assert attrs["bit_volts"] == pytest.approx(0.195)
        assert attrs["channel_name"] == "CH1"


def test_sample_index_file_is_chosen_by_gui_version(tmp_path, tree, gui_version):
    # Guards the 0.6 rename. Both layouts ship a sample-index file and a
    # seconds file; only the names swapped. Reading the wrong one yields
    # seconds where samples are expected, which is not a crash -- it is a
    # ~30000x error in the offset that nothing downstream would catch.
    #
    # The fixture writes the seconds file with the same underlying values, so
    # this assertion only holds if the correct file was read.
    tgt = run_oephys(tmp_path / "out.arf", tree)
    with arf.open_file(tgt, "r") as fp:
        offset = only_entry(fp)["CH1"].attrs["offset"]
    assert offset == pytest.approx(SAMPLE_OFFSET / rate(gui_version))
    # the seconds file would have given this instead
    assert offset != pytest.approx(SAMPLE_OFFSET / rate(gui_version) ** 2)


def test_recording_attributes_are_copied_to_entry(tmp_path, tree):
    tgt = run_oephys(tmp_path / "out.arf", tree)
    with arf.open_file(tgt, "r") as fp:
        entry = only_entry(fp)
        assert entry.attrs["GUI version"] in (OLD, NEW)
        assert "arfx-oephys" in entry.attrs["entry_creator"]


def test_string_events_become_marked_point_process(tmp_path, tree, gui_version):
    tgt = run_oephys(tmp_path / "out.arf", tree)
    with arf.open_file(tgt, "r") as fp:
        dset = only_entry(fp)[message_dset(gui_version)]
        assert arf.is_marked_pointproc(dset)
        assert dset["start"].tolist() == [100, 500, 900]
        assert dset["message"].tolist() == [b"start", b"mid", b"stop"]
        # the pre-0.6 layout carries an extra channel column
        expected = 3 if gui_version == OLD else 2
        assert len(dset.dtype.names) == expected
        assert len(dset.attrs["units"]) == expected


def test_event_metadata_is_dropped(tmp_path, tree, gui_version):
    tgt = run_oephys(tmp_path / "out.arf", tree)
    with arf.open_file(tgt, "r") as fp:
        dset = only_entry(fp)[message_dset(gui_version)]
        assert "event_metadata" not in dset.attrs


def test_int16_events(tmp_path, tmp_path_factory, gui_version):
    root = tmp_path_factory.mktemp("ttl")
    tree = build_tree(root, gui_version, events="int16")
    tgt = run_oephys(tmp_path / "out.arf", tree)
    with arf.open_file(tgt, "r") as fp:
        dset = only_entry(fp)[ttl_dset(gui_version)]
        assert dset["ttl"].tolist() == [1, -1, 1]


# ------------------------------------------------------- sample-offset fallback


def test_falls_back_to_sync_messages(tmp_path, tmp_path_factory, gui_version, caplog):
    # the sample index file can be deleted during spike sorting, so the start
    # time has to be recoverable from sync_messages.txt -- in either wording
    root = tmp_path_factory.mktemp("nosn")
    tree = build_tree(root, gui_version, write_sample_numbers=False)
    tgt = run_oephys(tmp_path / "out.arf", tree)
    with arf.open_file(tgt, "r") as fp:
        attrs = only_entry(fp)["CH1"].attrs
        assert attrs["offset"] == pytest.approx(SAMPLE_OFFSET / rate(gui_version))
    assert "falling back on sync_messages" in caplog.text


def structure_of(tree, gui_version):
    rec = recording_dir(tree, gui_version)
    return rec, json.loads((rec / "structure.oebin").read_text())["continuous"][0]


def test_find_sync_time_reads_both_wordings(tmp_path_factory, gui_version):
    # The matcher on its own. The two generations share no literal text, so
    # each pattern has to be exercised against the wording it was written for.
    tree = build_tree(tmp_path_factory.mktemp("syncfmt"), gui_version)
    rec, structure = structure_of(tree, gui_version)
    assert oephys.find_sync_time(rec, structure, Version(gui_version)) == SAMPLE_OFFSET


def test_find_sync_time_returns_none_for_unknown_processor(
    tmp_path_factory, gui_version
):
    tree = build_tree(tmp_path_factory.mktemp("syncmiss"), gui_version)
    rec, structure = structure_of(tree, gui_version)
    structure["source_processor_name"] = "Some Other Processor"
    assert oephys.find_sync_time(rec, structure, Version(gui_version)) is None


def test_find_sync_time_without_stream_identifier(tmp_path_factory):
    # a post-0.6 structure.oebin with no stream_name cannot be matched against
    # the newer wording, but that is a "no match", not a KeyError
    tree = build_tree(tmp_path_factory.mktemp("nostream"), NEW)
    rec, structure = structure_of(tree, NEW)
    del structure["stream_name"]
    assert oephys.find_sync_time(rec, structure, Version(NEW)) is None


def test_missing_offset_everywhere_is_fatal(tmp_path, tmp_path_factory, gui_version):
    root = tmp_path_factory.mktemp("noboth")
    tree = build_tree(
        root, gui_version, write_sample_numbers=False, write_sync_messages=False
    )
    with pytest.raises(FileNotFoundError):
        run_oephys(tmp_path / "out.arf", tree)


def test_sync_message_without_matching_processor_is_fatal(
    tmp_path, tmp_path_factory, gui_version
):
    root = tmp_path_factory.mktemp("nomatch")
    tree = build_tree(root, gui_version, write_sample_numbers=False)
    rec = recording_dir(tree, gui_version)
    (rec / "sync_messages.txt").write_text("nothing resembling a start time\n")
    with pytest.raises(RuntimeError, match="unable to determine sync time"):
        run_oephys(tmp_path / "out.arf", tree)


# ---------------------------------------------------------- unsupported inputs


def test_spikes_are_skipped_not_fatal(tmp_path, tmp_path_factory, gui_version, caplog):
    root = tmp_path_factory.mktemp("spikes")
    tree = build_tree(root, gui_version, spikes=True)
    tgt = run_oephys(tmp_path / "out.arf", tree)
    with arf.open_file(tgt, "r") as fp:
        # the continuous data still made it through
        assert "CH1" in only_entry(fp)
    assert "skipped" in caplog.text


def test_unsupported_event_type_is_skipped(
    tmp_path, tmp_path_factory, gui_version, caplog
):
    root = tmp_path_factory.mktemp("badevent")
    tree = build_tree(root, gui_version, events="float64")
    tgt = run_oephys(tmp_path / "out.arf", tree)
    with arf.open_file(tgt, "r") as fp:
        entry = only_entry(fp)
        assert "CH1" in entry
        assert message_dset(gui_version) not in entry
    assert "skipped" in caplog.text


def test_ragged_continuous_file_is_rejected(tmp_path, tmp_path_factory, gui_version):
    # a .dat whose length is not a multiple of the channel count means the
    # recording is truncated or the metadata is wrong; either way it cannot be
    # de-interleaved
    root = tmp_path_factory.mktemp("ragged")
    ragged = np.arange(NSAMPLES * NCHANNELS - 1, dtype="int16")
    tree = build_tree(root, gui_version, data=ragged)
    with pytest.raises(OSError, match="not a multiple of the channel count"):
        run_oephys(tmp_path / "out.arf", tree)


def test_channel_metadata_count_mismatch_warns(
    tmp_path, tmp_path_factory, gui_version, caplog
):
    root = tmp_path_factory.mktemp("mismatch")
    tree = build_tree(root, gui_version, channel_metadata_count=1)
    run_oephys(tmp_path / "out.arf", tree)
    assert "don't match" in caplog.text


def test_unparseable_directory_name_exits(tmp_path, tmp_path_factory, gui_version):
    root = tmp_path_factory.mktemp("badname")
    tree = build_tree(root, gui_version)
    renamed = tree.parent / "no_timestamp_here"
    tree.rename(renamed)
    with pytest.raises(SystemExit):
        run_oephys(tmp_path / "out.arf", renamed)


# ----------------------------------------------------------------- CLI options


def test_channel_list_filters_continuous(tmp_path, tree):
    channel_file = tmp_path / "channels.txt"
    channel_file.write_text("CH2\n")
    tgt = run_oephys(
        tmp_path / "out.arf", tree, extra=["--channel-list", str(channel_file)]
    )
    with arf.open_file(tgt, "r") as fp:
        entry = only_entry(fp)
        assert "CH2" in entry
        assert "CH1" not in entry


def test_extra_attributes_and_datatype(tmp_path, tree):
    tgt = run_oephys(
        tmp_path / "out.arf",
        tree,
        extra=["-k", "bird=C194", "-k", "pen=1", "-T", "EXTRAC_HP"],
    )
    with arf.open_file(tgt, "r") as fp:
        entry = only_entry(fp)
        assert entry.attrs["bird"] == "C194"
        # C194 is not valid JSON so it stays a string, while pen=1 is a number.
        # core.ParseKeyVal used to store both as strings.
        assert entry.attrs["pen"] == 1
        assert entry["CH1"].attrs["datatype"] == arf.DataTypes.EXTRAC_HP


def test_skip_empty(tmp_path, tmp_path_factory, gui_version, caplog):
    root = tmp_path_factory.mktemp("empty")
    tree = build_tree(
        root,
        gui_version,
        data=np.array([], dtype="int16"),
        write_sample_numbers=False,
    )
    tgt = run_oephys(tmp_path / "out.arf", tree, extra=["--skip-empty"])
    with arf.open_file(tgt, "r") as fp:
        assert "CH1" not in only_entry(fp)
    assert "skipping empty dataset" in caplog.text


def test_empty_recording_with_empty_index_falls_back(
    tmp_path, tmp_path_factory, gui_version, caplog
):
    # An aborted recording leaves a zero-length continuous.dat alongside a
    # zero-length sample index. An empty index is treated the same as a missing
    # one, so the start time comes from sync_messages and --skip-empty can then
    # do its job -- previously the IndexError fired while the recording was
    # being opened, before the flag was consulted.
    root = tmp_path_factory.mktemp("emptyidx")
    tree = build_tree(root, gui_version, data=np.array([], dtype="int16"))
    tgt = run_oephys(tmp_path / "out.arf", tree, extra=["--skip-empty"])
    with arf.open_file(tgt, "r") as fp:
        assert "CH1" not in only_entry(fp)
    assert "is empty" in caplog.text


def test_dry_run_writes_nothing(tmp_path, tree):
    tgt = tmp_path / "out.arf"
    oephys.script(["--dry-run", "-f", str(tgt), str(tree)])
    assert not tgt.exists()


def test_multiple_recordings_in_one_tree(tmp_path, tmp_path_factory, gui_version):
    root = tmp_path_factory.mktemp("multi")
    tree = build_tree(root, gui_version)
    # a second recording under the same experiment
    src = recording_dir(tree, gui_version)
    dst = src.parent / "recording2"
    import shutil

    shutil.copytree(src, dst)
    tgt = run_oephys(tmp_path / "out.arf", tree)
    with arf.open_file(tgt, "r") as fp:
        assert len([n for n in fp if arf.is_entry(fp[n])]) == 2


def test_appends_to_existing_file(tmp_path, tmp_path_factory, gui_version):
    root = tmp_path_factory.mktemp("append")
    tree = build_tree(root, gui_version)
    tgt = tmp_path / "out.arf"
    run_oephys(tgt, tree)
    other = build_tree(tmp_path_factory.mktemp("append2"), gui_version)
    run_oephys(tgt, other)
    with arf.open_file(tgt, "r") as fp:
        assert len([n for n in fp if arf.is_entry(fp[n])]) == 2
