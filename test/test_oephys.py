# -*- mode: python -*-
"""Tests for arfx.oephys (the arfx-oephys script).

open-ephys stores a recording as a directory tree whose layout changed at GUI
version 0.6.0, and oephys.py dispatches on the "GUI version" string in
structure.oebin to handle both. That dispatch is the module's main complexity,
so the fixture is parametrized over both generations and every test runs twice.

The tree is fabricated rather than checked in: a real recording is dominated by
continuous.dat and by per-sample index files that run to gigabytes, none of
which carry information the tests need.

The fabricated metadata is copied from a real GUI 1.0.2 recording rather than
invented, because the details that matter here are exactly the ones that are
easy to get wrong from reading the reader: folder naming, which index file
holds samples versus seconds, the channel metadata key set, and above all the
sync_messages.txt wording, which changed completely between generations.
"""

import datetime
import json

import arf
import numpy as np
import pytest
from packaging.version import Version

from arfx import oephys

TIMESTAMP_DIR = "P397_2026-06-17_11-08-34_arc6-main"
EXPECTED_TIME = datetime.datetime(2026, 6, 17, 11, 8, 34)
NSAMPLES = 1000
NCHANNELS = 2
SAMPLE_OFFSET = 152832

OLD, NEW = "0.5.5", "1.0.2"

# The two generations differ in more than the index filenames, so each is
# described by a profile taken from a real recording rather than by branching
# at every use. 0.5.x identifies a stream by a numeric sub-index; from 0.6 on
# it is a name, and source_processor_sub_idx is gone from structure.oebin
# altogether -- which is what breaks the sync_messages fallback.
PROFILES = {
    OLD: dict(
        record_node=103,
        processor_name="Rhythm FPGA",
        processor_id=100,
        sampling_rate=30000,  # an int here, a float from 0.6 on
        continuous_folder="Rhythm_FPGA-100.0/",
        ttl_folder="Rhythm_FPGA-100.0/TTL_1/",
        message_folder="Network_Events-104.0/TEXT_group_1/",
        sample_index="timestamps.npy",
        seconds_index="synchronized_timestamps.npy",
        ttl_states="channel_states.npy",
        channel_extra={"source_processor_index": 0, "recorded_processor_index": 0},
        channel_history="Rhythm FPGA -> Record Node",
        channel_identifier="genericdata.continuous",
    ),
    NEW: dict(
        record_node=107,
        processor_name="Acquisition Board",
        processor_id=100,
        stream_name="acquisition_board",
        sampling_rate=30000.0,
        continuous_folder="Acquisition_Board-100.acquisition_board/",
        ttl_folder="Acquisition_Board-100.acquisition_board/TTL/",
        message_folder="MessageCenter/",
        sample_index="sample_numbers.npy",
        seconds_index="timestamps.npy",
        ttl_states="states.npy",
        channel_extra={"type": 0},
        channel_history="Acquisition Board -> Record Node",
        channel_identifier="acq-board.rhythm.continuous.ephys",
    ),
}


def is_old(gui_version):
    return gui_version == OLD


def channel_metadata(gui_version, count):
    """Per-channel metadata, with the key set a real recording of this era carries."""
    p = PROFILES[gui_version]
    return [
        {
            "channel_name": f"CH{i + 1}",
            "description": "Headstage data channel",
            "identifier": p["channel_identifier"],
            "history": p["channel_history"],
            "bit_volts": 0.19499999284744263,
            "units": "uV",
            **p["channel_extra"],
        }
        for i in range(count)
    ]


def sync_messages_text(gui_version):
    """sync_messages.txt, verbatim in the wording this generation writes.

    The two generations share no literal text at all beyond the processor
    name. That is the whole problem: a reader written against one of them does
    not fail loudly against the other, it simply matches nothing.
    """
    p = PROFILES[gui_version]
    rate = int(p["sampling_rate"])
    if is_old(gui_version):
        return (
            "Software time: 153530@1000000Hz\n"
            f"Processor: {p['processor_name']} Id: {p['processor_id']} "
            f"subProcessor: 0 start time: {SAMPLE_OFFSET}@{rate}Hz\n"
        )
    return (
        "Software Time (milliseconds since midnight Jan 1st 1970 UTC): "
        "1781708914429\n"
        f"Start Time for {p['processor_name']} ({p['processor_id']}) - "
        f"{p['stream_name']} @ {rate} Hz: {SAMPLE_OFFSET}\n"
    )


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
    p = PROFILES[gui_version]
    rate = p["sampling_rate"]
    base = (
        root
        / TIMESTAMP_DIR
        / f"Record Node {p['record_node']}"
        / "experiment1"
        / "recording1"
    )
    base.mkdir(parents=True)
    event_folder = p["message_folder"] if events == "string" else p["ttl_folder"]

    def write_indices(directory, sample_numbers):
        """Write both index files the way the GUI of this generation would.

        Both layouts ship two index files, and the 0.6 rename swapped which
        name means what: `sample_numbers.npy` (int64 sample indices) took over
        the name `timestamps.npy` had, and `timestamps.npy` took over
        `synchronized_timestamps.npy`, holding float64 seconds. Writing both
        means a dispatch that reads the wrong one gets seconds where it wanted
        samples, instead of a missing file.
        """
        sample_numbers = np.asarray(sample_numbers, dtype="int64")
        np.save(directory / p["sample_index"], sample_numbers)
        np.save(directory / p["seconds_index"], sample_numbers / rate)

    # --- continuous
    cont_dir = base / "continuous" / p["continuous_folder"].strip("/")
    cont_dir.mkdir(parents=True)
    if data is None:
        data = continuous_ramp()
    data.tofile(cont_dir / "continuous.dat")
    if write_sample_numbers:
        write_indices(cont_dir, np.arange(SAMPLE_OFFSET, SAMPLE_OFFSET + len(data)))
    else:
        # spike sorting deletes the sample index but leaves the seconds file
        np.save(cont_dir / p["seconds_index"], np.arange(len(data)) / rate)

    nmeta = nchannels if channel_metadata_count is None else channel_metadata_count
    processor = dict(
        source_processor_name=p["processor_name"],
        source_processor_id=p["processor_id"],
        recorded_processor="Record Node",
        recorded_processor_id=p["record_node"],
    )
    if is_old(gui_version):
        processor["source_processor_sub_idx"] = 0
    else:
        processor["stream_name"] = p["stream_name"]

    structure = {
        "GUI version": gui_version,
        "continuous": [
            dict(
                folder_name=p["continuous_folder"],
                sample_rate=rate,
                num_channels=nchannels,
                channels=channel_metadata(gui_version, nmeta),
                **processor,
            )
        ],
        "spikes": [],
    }

    # --- events
    if events is not None:
        event_dir = base / "events" / event_folder.strip("/")
        event_dir.mkdir(parents=True)
        write_indices(event_dir, [100, 500, 900])
        event_structure = dict(
            folder_name=event_folder,
            channel_name="Network messages" if events == "string" else "TTL input",
            description="Messages received through the network events module",
            identifier="external.network.rawData",
            sample_rate=rate,
            type=events,
            source_processor="Network Events",
        )
        if events == "string":
            # a real text.npy is fixed-width bytes, sized to the longest
            # message the GUI allows rather than to the content
            np.save(
                event_dir / "text.npy",
                np.array([b"start", b"mid", b"stop"], dtype="|S513"),
            )
            if is_old(gui_version):
                np.save(event_dir / "channels.npy", np.array([0, 0, 0]))
                # a list of per-field descriptors, not a scalar; it has to be
                # dropped before the attrs reach h5py
                event_structure["event_metadata"] = [
                    {"name": "Text", "description": "Message text", "type": "string"}
                ]
        elif events == "int16":
            np.save(event_dir / p["ttl_states"], np.array([1, -1, 1], dtype="int16"))
            # not read by arfx, but the GUI always writes it
            np.save(event_dir / "full_words.npy", np.array([1, 0, 1], dtype="uint64"))
        if is_old(gui_version):
            event_structure["num_channels"] = 8
        structure["events"] = [event_structure]

    if spikes:
        structure["spikes"] = [
            dict(folder_name="Spike_Detector-105.0/", sample_rate=rate)
        ]

    (base / "structure.oebin").write_text(json.dumps(structure))

    if write_sync_messages:
        (base / "sync_messages.txt").write_text(sync_messages_text(gui_version))

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


def rate(gui_version):
    return PROFILES[gui_version]["sampling_rate"]


def message_dset(gui_version):
    """The dataset name oephys derives from the message-event folder."""
    return PROFILES[gui_version]["message_folder"].strip("/").replace("/", "_")


def ttl_dset(gui_version):
    return PROFILES[gui_version]["ttl_folder"].strip("/").replace("/", "_")


def recording_dir(tree, gui_version):
    node = PROFILES[gui_version]["record_node"]
    return tree / f"Record Node {node}" / "experiment1" / "recording1"


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
        # core.ParseKeyVal does no type coercion, so numeric metadata is stored
        # as a string. -k pen=1 is not readable back as an integer.
        assert entry.attrs["pen"] == "1"
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
