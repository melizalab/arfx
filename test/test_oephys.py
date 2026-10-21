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

import datetime
import json

import arf
import numpy as np
import pytest
from conftest import (
    EXPECTED_SOFTWARE_TIME,
    EXPECTED_TIME,
    NCHANNELS,
    NEW,
    NSAMPLES,
    OLD,
    SAMPLE_OFFSET,
    SOFTWARE_TIME_MS,
    TIMESTAMP_DIR,
    build_tree,
    is_old,
    message_dset,
    only_entry,
    rate,
    recording_dir,
    run_oephys,
    ttl_dset,
)
from packaging.version import Version

from arfx import nbank, oephys

# ------------------------------------------------------------------- happy path


def test_creates_one_entry_per_recording(tmp_path, tree, gui_version):
    # The timestamp comes from sync_messages.txt where the GUI writes a
    # wall-clock time there, and from the session directory name otherwise --
    # the pre-0.6 "Software time:" line is a monotonic clock, not an epoch.
    expected = EXPECTED_TIME if is_old(gui_version) else EXPECTED_SOFTWARE_TIME
    tgt = run_oephys(tmp_path / "out.arf", tree)
    with arf.open_file(tgt, "r") as fp:
        entry = only_entry(fp)
        assert arf.timestamp_to_datetime(entry.attrs["timestamp"]) == expected


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


# ------------------------------------------------- discontinuous sample numbers


def jumpy_sample_numbers(n=NSAMPLES):
    """The shape of the counter in a real recording (C182_4_1, issue #47).

    The board's counter ran on, jumped back two seconds, then reset to a small
    number while the signal was still startup padding, and counted on without a
    break to the end. Returns the index and the first sample number of the
    final run, which is where the offset has to come from.
    """
    first_break, second_break, final_start = 200, 600, 300
    runs = (
        np.arange(89523584, 89523584 + first_break),
        np.arange(89475872, 89475872 + second_break - first_break),
        np.arange(final_start, final_start + n - second_break),
    )
    return np.concatenate(runs), final_start - second_break


def test_offset_comes_from_the_final_contiguous_run(
    tmp_path, tmp_path_factory, gui_version, caplog
):
    # Events are stamped on the clock in force when they happened, so a
    # reset during startup means the first sample has to be given the
    # number it would have on the final clock -- here a negative one.
    sample_numbers, first = jumpy_sample_numbers()
    tree = build_tree(
        tmp_path_factory.mktemp("jumpy"), gui_version, sample_numbers=sample_numbers
    )
    tgt = run_oephys(tmp_path / "out.arf", tree)
    with arf.open_file(tgt, "r") as fp:
        entry = only_entry(fp)
        assert entry["CH1"].attrs["offset"] == pytest.approx(first / rate(gui_version))
        assert entry["CH1"].attrs["offset"] < 0
        # the data themselves are untouched, whatever the clock did
        assert entry["CH1"][0] == 0
        assert entry["CH1"][-1] == NSAMPLES - 1


def test_discontinuities_are_reported_with_index_and_size(
    tmp_path, tmp_path_factory, gui_version, caplog
):
    sample_numbers, _ = jumpy_sample_numbers()
    tree = build_tree(
        tmp_path_factory.mktemp("jumpywarn"), gui_version, sample_numbers=sample_numbers
    )
    run_oephys(tmp_path / "out.arf", tree)
    assert "jump at index 200 (89523783 -> 89475872, -47911)" in caplog.text
    assert "jump at index 600 (89476271 -> 300, -89475971)" in caplog.text
    assert "final contiguous run" in caplog.text


def test_contiguous_sample_numbers_are_not_flagged(tmp_path, tree, caplog):
    run_oephys(tmp_path / "out.arf", tree)
    assert "jump" not in caplog.text
    assert "sample numbers but" not in caplog.text


def test_sample_number_count_mismatch_is_flagged(
    tmp_path, tmp_path_factory, gui_version, caplog
):
    tree = build_tree(
        tmp_path_factory.mktemp("short"),
        gui_version,
        sample_numbers=np.arange(SAMPLE_OFFSET, SAMPLE_OFFSET + NSAMPLES - 10),
    )
    run_oephys(tmp_path / "out.arf", tree)
    assert f"{NSAMPLES - 10} sample numbers but {NSAMPLES} samples" in caplog.text


def test_events_stepping_backward_across_a_clock_reset_are_flagged(
    tmp_path, tmp_path_factory, gui_version, caplog
):
    # the first message of C182_4_1 carried the pre-reset clock
    tree = build_tree(
        tmp_path_factory.mktemp("evreset"),
        gui_version,
        event_samples=(89521870, 48367, 140080),
    )
    run_oephys(tmp_path / "out.arf", tree)
    assert "step backward after row 0 (89521870 -> 48367)" in caplog.text


def test_small_event_reordering_is_not_flagged(
    tmp_path, tmp_path_factory, gui_version, caplog
):
    # network messages can arrive a millisecond out of order
    tree = build_tree(
        tmp_path_factory.mktemp("evjitter"),
        gui_version,
        event_samples=(1000, 970, 5000),
    )
    run_oephys(tmp_path / "out.arf", tree)
    assert "step backward" not in caplog.text


def test_ordered_events_are_not_flagged(tmp_path, tree, caplog):
    run_oephys(tmp_path / "out.arf", tree)
    assert "step backward" not in caplog.text


def test_find_discontinuities_spans_chunks():
    # a jump that lands on a chunk boundary must not be missed or doubled
    x = np.concatenate([np.arange(10), np.arange(100, 110)])
    for chunk in (3, 5, 9, 10, 100):
        assert oephys.find_discontinuities(x, chunk=chunk) == [(9, 9, 100)]


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


def undated(tree):
    renamed = tree.parent / "no_timestamp_here"
    tree.rename(renamed)
    return renamed


def test_undated_directory_needs_no_name_when_sync_messages_has_the_time(
    tmp_path, tmp_path_factory
):
    tree = undated(build_tree(tmp_path_factory.mktemp("badname"), NEW))
    tgt = run_oephys(tmp_path / "out.arf", tree)
    with arf.open_file(tgt, "r") as fp:
        entry = only_entry(fp)
        assert arf.timestamp_to_datetime(entry.attrs["timestamp"]) == (
            EXPECTED_SOFTWARE_TIME
        )


def test_undated_directory_without_wall_clock_time_exits_and_writes_nothing(
    tmp_path, tmp_path_factory
):
    tree = undated(build_tree(tmp_path_factory.mktemp("badname"), OLD))
    with pytest.raises(SystemExit):
        run_oephys(tmp_path / "out.arf", tree)
    assert not (tmp_path / "out.arf").exists()


def test_session_time_stands_in_for_the_directory_name(tmp_path, tmp_path_factory):
    tree = undated(build_tree(tmp_path_factory.mktemp("badname"), OLD))
    tgt = run_oephys(
        tmp_path / "out.arf", tree, extra=["--session-time", "2026-10-12T09:30:00"]
    )
    with arf.open_file(tgt, "r") as fp:
        stamp = arf.timestamp_to_datetime(only_entry(fp).attrs["timestamp"])
    assert stamp == datetime.datetime(2026, 10, 12, 9, 30, 0)


def test_session_time_does_not_override_the_wall_clock_time(tmp_path, tmp_path_factory):
    tree = build_tree(tmp_path_factory.mktemp("wallclock"), NEW)
    tgt = run_oephys(
        tmp_path / "out.arf", tree, extra=["--session-time", "2000-01-01T00:00:00"]
    )
    with arf.open_file(tgt, "r") as fp:
        stamp = arf.timestamp_to_datetime(only_entry(fp).attrs["timestamp"])
    assert stamp == EXPECTED_SOFTWARE_TIME


def test_find_session_time_takes_the_innermost_timestamp(tmp_path):
    path = tmp_path / "2020-01-01_00-00-00" / "c182_2026-10-12_12-30-45"
    assert oephys.find_session_time(path) == datetime.datetime(2026, 10, 12, 12, 30, 45)


def test_find_session_time_rejects_an_impossible_date(tmp_path):
    assert oephys.find_session_time(tmp_path / "x_2026-13-45_25-61-61") is None


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


ANIMAL = "b1360fc4-34b8-41cb-a7c7-ec5d774238e9"


def test_experimenter_and_animal(tmp_path, tree):
    """--experimenter and --animal are stored as strings and listed with the -k
    attributes, so the neurobank reader registers them; the animal is a UUID in
    its canonical form."""
    tgt = run_oephys(
        tmp_path / "out.arf",
        tree,
        extra=["--experimenter", "1234", "--animal", ANIMAL.upper(), "-k", "pen=1"],
    )
    with arf.open_file(tgt, "r") as fp:
        entry = only_entry(fp)
        assert entry.attrs["experimenter"] == "1234"
        assert entry.attrs["animal"] == ANIMAL
        assert sorted(entry.attrs["user_attributes"]) == [
            "animal",
            "experimenter",
            "pen",
        ]


def test_animal_not_a_uuid_is_a_warning(tmp_path, tree, caplog):
    """A test recording may have no real animal; it just can't be deposited."""
    tgt = run_oephys(tmp_path / "out.arf", tree, extra=["--animal", "C194"])
    assert "not a full identifier (UUID)" in caplog.text
    with arf.open_file(tgt, "r") as fp:
        assert only_entry(fp).attrs["animal"] == "C194"


@pytest.mark.parametrize(
    "extra",
    [
        ["--animal", ANIMAL, "-k", f"animal={ANIMAL}"],
        ["--animal", ANIMAL, "-k", "bird=C194"],
        ["--experimenter", "a", "-k", "experimenter=a"],
    ],
)
def test_option_and_k_for_the_same_field(tmp_path, tree, extra):
    tgt = tmp_path / "out.arf"
    with pytest.raises(SystemExit):
        run_oephys(tgt, tree, extra=extra)
    assert not tgt.exists()


# what oeaudio-present logged for P314_1_6
PRESENTER = {
    "animal": "P314",
    "experimenter": "tbyron",
    "experiment": "selectivity-chorus-short",
    "hemisphere": "L",
    "pen": 1,
    "site": 4,
    "x": 893.4,
    "y": -749.5,
    "z": -2500,
}


def presenter_tree(tmp_path_factory, gui_version, metadata=PRESENTER, text=None):
    if text is None:
        text = "metadata: " + json.dumps(metadata)
    root = tmp_path_factory.mktemp("presenter")
    return build_tree(root, gui_version, messages=(b"start", text.encode(), b"stop"))


def test_presenter_metadata_becomes_attributes(tmp_path, tmp_path_factory, gui_version):
    """Under the registry's names, without the presenter's animal (a band
    name), and listed so that the neurobank reader registers them."""
    tree = presenter_tree(tmp_path_factory, gui_version)
    tgt = run_oephys(tmp_path / "out.arf", tree, extra=["--animal", ANIMAL])
    expected = {
        "animal": ANIMAL,
        "experimenter": "tbyron",
        "protocol": "selectivity-chorus-short",
        "hemisphere": "L",
        "pen": 1,
        "site": 4,
        "x": 893.4,
        "y": -749.5,
        "z": -2500,
    }
    with arf.open_file(tgt, "r") as fp:
        entry = only_entry(fp)
        assert {k: entry.attrs[k] for k in expected} == expected
        assert "experiment" not in entry.attrs
        assert sorted(entry.attrs["user_attributes"]) == sorted(expected)
    metadata = nbank.read_metadata(tgt)
    assert {k: metadata[k] for k in expected} == expected


def test_given_attributes_matching_the_presenter(tmp_path, tmp_path_factory):
    tree = presenter_tree(tmp_path_factory, NEW)
    extra = ["--experimenter", "tbyron", "-k", "pen=1", "-k", "hemisphere=L"]
    tgt = run_oephys(tmp_path / "out.arf", tree, extra=extra)
    with arf.open_file(tgt, "r") as fp:
        assert only_entry(fp).attrs["pen"] == 1


@pytest.mark.parametrize(
    "extra",
    [
        ["-k", "pen=unknown"],
        ["-k", "pen=true"],
        ["--experimenter", "someone"],
        # compared under the registry's name for it
        ["-k", "experiment=other"],
    ],
)
def test_given_attributes_differing_from_the_presenter(
    tmp_path, tmp_path_factory, caplog, extra
):
    tree = presenter_tree(tmp_path_factory, NEW)
    tgt = tmp_path / "out.arf"
    with pytest.raises(SystemExit):
        run_oephys(tgt, tree, extra=extra)
    assert "--override-presenter-metadata" in caplog.text
    assert not tgt.exists()


def test_override_presenter_metadata(tmp_path, tmp_path_factory, caplog):
    tree = presenter_tree(tmp_path_factory, NEW)
    extra = ["--override-presenter-metadata", "-k", "pen=2"]
    tgt = run_oephys(tmp_path / "out.arf", tree, extra=extra)
    with arf.open_file(tgt, "r") as fp:
        entry = only_entry(fp)
        assert entry.attrs["pen"] == 2
        assert entry.attrs["site"] == 4
    assert "using 2 instead of 1" in caplog.text


def test_unreadable_presenter_metadata_is_a_warning(tmp_path, tmp_path_factory, caplog):
    tree = presenter_tree(tmp_path_factory, NEW, text='metadata: {"pen": 1,}')
    tgt = run_oephys(tmp_path / "out.arf", tree, extra=["-k", "pen=2"])
    assert "unreadable metadata message" in caplog.text
    with arf.open_file(tgt, "r") as fp:
        assert only_entry(fp).attrs["pen"] == 2


def test_presenter_metadata_that_cant_be_stored_is_skipped(
    tmp_path, tmp_path_factory, caplog
):
    tree = presenter_tree(tmp_path_factory, NEW, metadata={"pen": 1, "pos": [1, 2]})
    tgt = run_oephys(tmp_path / "out.arf", tree)
    assert "can't be stored" in caplog.text
    with arf.open_file(tgt, "r") as fp:
        entry = only_entry(fp)
        assert entry.attrs["pen"] == 1
        assert "pos" not in entry.attrs


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
    # two sessions, so two distinct entry names -- which is what two real
    # recordings would have, since the session directory carries the timestamp
    root = tmp_path_factory.mktemp("append")
    tree = build_tree(root, gui_version)
    tgt = tmp_path / "out.arf"
    run_oephys(tgt, tree)
    other = build_tree(
        tmp_path_factory.mktemp("append2"),
        gui_version,
        session="P397_2026-06-17_15-30-00_arc6-main",
    )
    run_oephys(tgt, other)
    with arf.open_file(tgt, "r") as fp:
        assert len([n for n in fp if arf.is_entry(fp[n])]) == 2


def test_entry_name_does_not_depend_on_where_the_tree_lives(
    tmp_path, tmp_path_factory, gui_version
):
    # The name used to be str(dir) with the separators replaced, so the same
    # recording imported from a different machine, mount point or working
    # directory produced a different entry name for identical data -- and two
    # such archives could not be merged without duplicates.
    names = []
    for label in ("here", "somewhere/else/entirely"):
        root = tmp_path_factory.mktemp("loc") / label
        root.mkdir(parents=True)
        tgt = tmp_path / f"{root.name}.arf"
        run_oephys(tgt, build_tree(root, gui_version))
        with arf.open_file(tgt, "r") as fp:
            names.append([n for n in fp if arf.is_entry(fp[n])])
    assert names[0] == names[1]
    assert names[0][0].startswith("P397_2026-06-17_11-08-34_arc6-main_Record Node ")


def test_reimporting_the_same_recording_is_refused(tmp_path, gui_version, caplog):
    # the other side of a reproducible name: a repeat import collides instead
    # of silently adding a second copy of the same data
    tree = build_tree(tmp_path / "src", gui_version)
    tgt = tmp_path / "out.arf"
    run_oephys(tgt, tree)
    with pytest.raises(SystemExit) as excinfo:
        run_oephys(tgt, tree)
    assert excinfo.value.code == 1
    assert "is already in" in caplog.text
    with arf.open_file(tgt, "r") as fp:
        assert len([n for n in fp if arf.is_entry(fp[n])]) == 1


def test_failed_import_leaves_no_partial_entry(
    tmp_path, tmp_path_factory, gui_version, monkeypatch
):
    # otherwise the duplicate check would skip the broken entry on the next run
    def fail(self, nsamples):
        raise OSError("disk went away")
        yield

    tree = build_tree(tmp_path_factory.mktemp("fail"), gui_version)
    tgt = tmp_path / "out.arf"
    monkeypatch.setattr(oephys.continuous_dset, "iter_chunks", fail)
    with pytest.raises(OSError, match="disk went away"):
        run_oephys(tgt, tree)
    monkeypatch.undo()
    with arf.open_file(tgt, "r") as fp:
        assert [n for n in fp if arf.is_entry(fp[n])] == []
    run_oephys(tgt, tree)
    with arf.open_file(tgt, "r") as fp:
        assert len([n for n in fp if arf.is_entry(fp[n])]) == 1


def test_appending_a_recording_with_other_channels_is_refused(
    tmp_path, tmp_path_factory, gui_version, caplog
):
    tgt = tmp_path / "out.arf"
    run_oephys(
        tgt,
        build_tree(tmp_path_factory.mktemp("a"), gui_version, session=TIMESTAMP_DIR),
    )
    narrow = build_tree(
        tmp_path_factory.mktemp("b"),
        gui_version,
        nchannels=1,
        data=np.zeros((NSAMPLES, 1), dtype="int16"),
        session="C_2026-10-12_12-00-00",
    )
    with pytest.raises(SystemExit):
        run_oephys(tgt, narrow)
    assert "channels differ from entry" in caplog.text
    assert "(1 missing, 0 new; e.g. CH2)" in caplog.text
    with arf.open_file(tgt, "r") as fp:
        assert len([n for n in fp if arf.is_entry(fp[n])]) == 1


def test_appending_a_recording_with_another_sampling_rate_is_refused(
    tmp_path, tmp_path_factory, gui_version, caplog
):
    tgt = tmp_path / "out.arf"
    run_oephys(tgt, build_tree(tmp_path_factory.mktemp("a"), gui_version))
    other = build_tree(
        tmp_path_factory.mktemp("b"),
        gui_version,
        sampling_rate=rate(gui_version) / 2,
        session="C_2026-10-12_12-00-00",
    )
    with pytest.raises(SystemExit):
        run_oephys(tgt, other)
    assert "sampling rate differs from entry" in caplog.text
    with arf.open_file(tgt, "r") as fp:
        assert len([n for n in fp if arf.is_entry(fp[n])]) == 1


def test_appending_a_matching_recording_does_not_warn(
    tmp_path, tmp_path_factory, gui_version, caplog
):
    tgt = tmp_path / "out.arf"
    run_oephys(tgt, build_tree(tmp_path_factory.mktemp("a"), gui_version))
    run_oephys(
        tgt,
        build_tree(
            tmp_path_factory.mktemp("b"), gui_version, session="C_2026-10-12_12-00-00"
        ),
    )
    assert "differs from entry" not in caplog.text


# ------------------------------------------- several recordings in one session


def test_every_recording_in_a_session_becomes_an_entry(tmp_path, tmp_path_factory):
    # Newer versions of the GUI write several experiments, and several
    # recordings per experiment, into one session directory. Each one is its
    # own entry, exactly as it would be if it had its own folder.
    root = tmp_path_factory.mktemp("multi")
    for experiment, recording in ((1, 1), (1, 2), (2, 1)):
        build_tree(root, NEW, experiment=experiment, recording=recording)
    tree = root / TIMESTAMP_DIR
    tgt = run_oephys(tmp_path / "out.arf", tree)
    with arf.open_file(tgt, "r") as fp:
        names = [n for n in arf.keys_by_creation(fp) if arf.is_entry(fp[n])]
    assert len(names) == 3
    assert [n.rsplit("_", 2)[-2:] for n in names] == [
        ["experiment1", "recording1"],
        ["experiment1", "recording2"],
        ["experiment2", "recording1"],
    ]


def test_recordings_in_a_session_get_their_own_timestamps(tmp_path, tmp_path_factory):
    # The whole point of #44: the session directory name gives the start of the
    # first recording only, so taking every entry's timestamp from it put three
    # recordings made minutes apart at the same instant.
    root = tmp_path_factory.mktemp("multitime")
    offsets = (0, 8128, 94298)  # ms, from examples/double_2026-08-10_10-07-17_double
    for (experiment, recording), offset in zip(
        ((1, 1), (1, 2), (2, 1)), offsets, strict=True
    ):
        build_tree(
            root,
            NEW,
            experiment=experiment,
            recording=recording,
            software_time_ms=SOFTWARE_TIME_MS + offset,
        )
    tgt = run_oephys(tmp_path / "out.arf", root / TIMESTAMP_DIR)
    with arf.open_file(tgt, "r") as fp:
        times = [
            arf.timestamp_to_datetime(fp[n].attrs["timestamp"])
            for n in arf.keys_by_creation(fp)
            if arf.is_entry(fp[n])
        ]
    assert times == [
        EXPECTED_SOFTWARE_TIME + datetime.timedelta(milliseconds=offset)
        for offset in offsets
    ]
    assert len(set(times)) == 3


def test_recordings_are_ordered_naturally(tmp_path, tmp_path_factory):
    # glob order is filesystem-dependent, and plain lexicographic sorting puts
    # recording10 before recording2
    root = tmp_path_factory.mktemp("natsort")
    for recording in (1, 2, 10):
        build_tree(root, NEW, recording=recording)
    tgt = run_oephys(tmp_path / "out.arf", root / TIMESTAMP_DIR)
    with arf.open_file(tgt, "r") as fp:
        names = [n for n in arf.keys_by_creation(fp) if arf.is_entry(fp[n])]
    assert [n.rsplit("_", 1)[-1] for n in names] == [
        "recording1",
        "recording2",
        "recording10",
    ]


def test_pre_0_6_recording_falls_back_to_the_directory_timestamp(
    tmp_path, tmp_path_factory, caplog
):
    # The pre-0.6 "Software time: 153530@1000000Hz" line is a monotonic clock,
    # not an epoch, so there is nothing to take a wall-clock time from and the
    # entry keeps the session's. Using that line would date it to 1970.
    #
    # These sessions hold a single recording, so the fallback is exact rather
    # than merely the best available -- the directory name is that recording's
    # start time. Multiple recordings per folder only arrive with 0.6.
    tree = build_tree(tmp_path_factory.mktemp("oldsession"), OLD)
    tgt = run_oephys(tmp_path / "out.arf", tree, extra=["-v"])
    with arf.open_file(tgt, "r") as fp:
        entry = only_entry(fp)
        assert arf.timestamp_to_datetime(entry.attrs["timestamp"]) == EXPECTED_TIME
    assert "no wall-clock time" in caplog.text


def test_find_software_time_ignores_the_legacy_wording(tmp_path_factory):
    tree = build_tree(tmp_path_factory.mktemp("legacy"), OLD)
    assert oephys.find_software_time(recording_dir(tree, OLD)) is None


def test_find_software_time_without_sync_messages(tmp_path_factory):
    tree = build_tree(tmp_path_factory.mktemp("nosync"), NEW, write_sync_messages=False)
    assert oephys.find_software_time(recording_dir(tree, NEW)) is None
