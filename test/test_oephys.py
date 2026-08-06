# -*- mode: python -*-
"""Tests for arfx.oephys (the arfx-oephys script).

open-ephys stores a recording as a directory tree whose layout changed at GUI
version 0.6.0, and oephys.py dispatches on the "GUI version" string in
structure.oebin to handle both. That dispatch is the module's main complexity,
so the fixture is parametrized over both layouts and every test runs twice.

The tree is fabricated rather than checked in: it is a few JSON keys, some .npy
index files and a raw interleaved .dat, all of which are cheaper to generate
than to store.
"""

import datetime
import json

import arf
import numpy as np
import pytest

from arfx import oephys

TIMESTAMP_DIR = "rec_2023-10-16_16-30-54"
EXPECTED_TIME = datetime.datetime(2023, 10, 16, 16, 30, 54)
SAMPLING_RATE = 30000.0
NSAMPLES = 1000
NCHANNELS = 2
SAMPLE_OFFSET = 300
CONTINUOUS_FOLDER = "Rhythm_FPGA-100.0/"
EVENT_FOLDER = "Message_Center-904.0/TEXT1/"
PROCESSOR = dict(
    source_processor_name="Rhythm FPGA",
    source_processor_id=100,
    source_processor_sub_idx=0,
)

OLD, NEW = "0.5.5", "0.6.0"


def continuous_ramp():
    """Interleaved int16 data where channel j holds j*NSAMPLES + i."""
    data = np.empty((NSAMPLES, NCHANNELS), dtype="int16")
    for j in range(NCHANNELS):
        data[:, j] = np.arange(NSAMPLES) + j * NSAMPLES
    return data


def build_tree(
    root,
    gui_version=NEW,
    *,
    nchannels=NCHANNELS,
    data=None,
    write_sample_numbers=True,
    write_sync_messages=True,
    events="string",
    spikes=False,
    channel_metadata_count=None,
):
    """Fabricate an open-ephys binary-format recording directory.

    Returns the top-level path, which is what the script takes on the command
    line. The timestamp is parsed out of that directory's name, so it must keep
    the YYYY-MM-DD_HH-MM-SS form.
    """
    base = root / TIMESTAMP_DIR / "Record Node 101" / "experiment1" / "recording1"
    base.mkdir(parents=True)

    def write_indices(directory, sample_numbers):
        """Write both index files the way the GUI of this version would.

        Both layouts ship two index files, and the 0.6 rename swapped which
        name means what: `sample_numbers.npy` (int64 sample indices) took over
        the name `timestamps.npy` had, and `timestamps.npy` took over
        `synchronized_timestamps.npy`, holding float64 seconds. Writing both
        means a dispatch that reads the wrong one gets seconds where it wanted
        samples, instead of a missing file.
        """
        sample_numbers = np.asarray(sample_numbers, dtype="int64")
        seconds = sample_numbers / SAMPLING_RATE
        if gui_version == OLD:
            np.save(directory / "timestamps.npy", sample_numbers)
            np.save(directory / "synchronized_timestamps.npy", seconds)
        else:
            np.save(directory / "sample_numbers.npy", sample_numbers)
            np.save(directory / "timestamps.npy", seconds)

    # --- continuous
    cont_dir = base / "continuous" / CONTINUOUS_FOLDER.strip("/")
    cont_dir.mkdir(parents=True)
    if data is None:
        data = continuous_ramp()
    data.tofile(cont_dir / "continuous.dat")
    if write_sample_numbers:
        write_indices(cont_dir, np.arange(SAMPLE_OFFSET, SAMPLE_OFFSET + len(data)))
    else:
        # spike sorting deletes the sample index but leaves the seconds file
        seconds = np.arange(len(data)) / SAMPLING_RATE
        np.save(
            cont_dir
            / (
                "synchronized_timestamps.npy"
                if gui_version == OLD
                else "timestamps.npy"
            ),
            seconds,
        )

    nmeta = nchannels if channel_metadata_count is None else channel_metadata_count
    structure = {
        "GUI version": gui_version,
        "continuous": [
            dict(
                folder_name=CONTINUOUS_FOLDER,
                sample_rate=SAMPLING_RATE,
                num_channels=nchannels,
                channels=[
                    {"channel_name": f"CH{i + 1}", "bit_volts": 0.195}
                    for i in range(nmeta)
                ],
                **PROCESSOR,
            )
        ],
    }

    # --- events
    if events is not None:
        event_dir = base / "events" / EVENT_FOLDER.strip("/")
        event_dir.mkdir(parents=True)
        write_indices(event_dir, [100, 500, 900])
        if events == "string":
            np.save(event_dir / "text.npy", np.array([b"start", b"mid", b"stop"]))
            if gui_version == OLD:
                np.save(event_dir / "channels.npy", np.array([0, 0, 0]))
        elif events == "int16":
            state_name = "channel_states.npy" if gui_version == OLD else "states.npy"
            np.save(event_dir / state_name, np.array([1, -1, 1], dtype="int16"))
            # not read by arfx, but the GUI always writes it
            np.save(event_dir / "full_words.npy", np.array([1, 0, 1], dtype="int64"))
        structure["events"] = [
            dict(
                folder_name=EVENT_FOLDER,
                sample_rate=SAMPLING_RATE,
                type=events,
                event_metadata={"dropped": "on purpose"},
            )
        ]

    if spikes:
        structure["spikes"] = [
            dict(folder_name="Spike_Detector-105.0/", sample_rate=SAMPLING_RATE)
        ]

    (base / "structure.oebin").write_text(json.dumps(structure))

    if write_sync_messages:
        (base / "sync_messages.txt").write_text(
            "Software time: 12345@1000Hz\n"
            f"Processor: {PROCESSOR['source_processor_name']} "
            f"Id: {PROCESSOR['source_processor_id']} "
            f"subProcessor: {PROCESSOR['source_processor_sub_idx']} "
            f"start time: {SAMPLE_OFFSET}@{int(SAMPLING_RATE)}Hz\n"
        )

    return root / TIMESTAMP_DIR


@pytest.fixture(params=[OLD, NEW], ids=["gui<0.6", "gui>=0.6"])
def gui_version(request):
    return request.param


@pytest.fixture
def tree(tmp_path, gui_version):
    return build_tree(tmp_path, gui_version)


def run_oephys(tgt, *paths, extra=()):
    oephys.script([*extra, "-f", str(tgt), *(str(p) for p in paths)])
    return tgt


def only_entry(fp):
    """The script names entries after the full source path, so look it up by count."""
    names = [n for n in fp if arf.is_entry(fp[n])]
    assert len(names) == 1
    return fp[names[0]]


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


def test_continuous_attributes(tmp_path, tree):
    tgt = run_oephys(tmp_path / "out.arf", tree)
    with arf.open_file(tgt, "r") as fp:
        attrs = only_entry(fp)["CH1"].attrs
        assert attrs["sampling_rate"] == SAMPLING_RATE
        # the offset is the first sample index converted to seconds
        assert attrs["offset"] == pytest.approx(SAMPLE_OFFSET / SAMPLING_RATE)
        assert attrs["bit_volts"] == pytest.approx(0.195)
        assert attrs["channel_name"] == "CH1"


def test_sample_index_file_is_chosen_by_gui_version(tmp_path, tree):
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
    assert offset == pytest.approx(SAMPLE_OFFSET / SAMPLING_RATE)
    # the seconds file would have given this instead
    assert offset != pytest.approx(SAMPLE_OFFSET / SAMPLING_RATE**2)


def test_recording_attributes_are_copied_to_entry(tmp_path, tree):
    tgt = run_oephys(tmp_path / "out.arf", tree)
    with arf.open_file(tgt, "r") as fp:
        entry = only_entry(fp)
        assert entry.attrs["GUI version"] in (OLD, NEW)
        assert "arfx-oephys" in entry.attrs["entry_creator"]


def test_string_events_become_marked_point_process(tmp_path, tree, gui_version):
    tgt = run_oephys(tmp_path / "out.arf", tree)
    with arf.open_file(tgt, "r") as fp:
        dset = only_entry(fp)["Message_Center-904.0_TEXT1"]
        assert arf.is_marked_pointproc(dset)
        assert dset["start"].tolist() == [100, 500, 900]
        assert dset["message"].tolist() == [b"start", b"mid", b"stop"]
        # the pre-0.6 layout carries an extra channel column
        expected = 3 if gui_version == OLD else 2
        assert len(dset.dtype.names) == expected
        assert len(dset.attrs["units"]) == expected


def test_event_metadata_is_dropped(tmp_path, tree):
    tgt = run_oephys(tmp_path / "out.arf", tree)
    with arf.open_file(tgt, "r") as fp:
        dset = only_entry(fp)["Message_Center-904.0_TEXT1"]
        assert "event_metadata" not in dset.attrs


def test_int16_events(tmp_path, tmp_path_factory, gui_version):
    root = tmp_path_factory.mktemp("ttl")
    tree = build_tree(root, gui_version, events="int16")
    tgt = run_oephys(tmp_path / "out.arf", tree)
    with arf.open_file(tgt, "r") as fp:
        dset = only_entry(fp)["Message_Center-904.0_TEXT1"]
        assert dset["ttl"].tolist() == [1, -1, 1]


# ------------------------------------------------------- sample-offset fallback


def test_falls_back_to_sync_messages(tmp_path, tmp_path_factory, gui_version, caplog):
    # the sample index file is routinely deleted during spike sorting, so the
    # start time has to be recoverable from sync_messages.txt
    root = tmp_path_factory.mktemp("nosn")
    tree = build_tree(root, gui_version, write_sample_numbers=False)
    tgt = run_oephys(tmp_path / "out.arf", tree)
    with arf.open_file(tgt, "r") as fp:
        attrs = only_entry(fp)["CH1"].attrs
        assert attrs["offset"] == pytest.approx(SAMPLE_OFFSET / SAMPLING_RATE)
    assert "falling back on sync_messages" in caplog.text


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
    struct = tree / "Record Node 101" / "experiment1" / "recording1"
    (struct / "sync_messages.txt").write_text("Processor: Other Id: 1 nonsense\n")
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
        assert "Message_Center-904.0_TEXT1" not in entry
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
        # core.ParseKeyVal does no type coercion, so numeric metadata is stored
        # as a string. -k pen=1 is not readable back as an integer.
        assert entry.attrs["pen"] == "1"
        assert entry["CH1"].attrs["datatype"] == arf.DataTypes.EXTRAC_HP


def test_skip_empty(tmp_path, tmp_path_factory, gui_version, caplog):
    # the sample index is taken from sync_messages here; see
    # test_empty_recording_with_index_file_crashes for why it cannot come from
    # the .npy in this case
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


def test_empty_recording_with_index_file_crashes(
    tmp_path, tmp_path_factory, gui_version
):
    # CHARACTERIZATION: an aborted recording leaves a zero-length continuous.dat
    # alongside a zero-length sample index. continuous_dset reads the start time
    # as timestamps[0] without checking that the array is non-empty, so the
    # IndexError happens while the recording is being opened -- before
    # --skip-empty gets a chance to skip anything. The flag cannot rescue the
    # case it most obviously exists for.
    root = tmp_path_factory.mktemp("emptyidx")
    tree = build_tree(root, gui_version, data=np.array([], dtype="int16"))
    with pytest.raises(IndexError):
        run_oephys(tmp_path / "out.arf", tree, extra=["--skip-empty"])


def test_dry_run_writes_nothing(tmp_path, tree):
    tgt = tmp_path / "out.arf"
    oephys.script(["--dry-run", "-f", str(tgt), str(tree)])
    assert not tgt.exists()


def test_multiple_recordings_in_one_tree(tmp_path, tmp_path_factory, gui_version):
    root = tmp_path_factory.mktemp("multi")
    tree = build_tree(root, gui_version)
    # a second recording under the same experiment
    src = tree / "Record Node 101" / "experiment1" / "recording1"
    dst = tree / "Record Node 101" / "experiment1" / "recording2"
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
