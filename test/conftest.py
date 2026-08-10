# -*- mode: python -*-
"""Fixtures shared across the arfx test suite.

Two families live here: synthetic ARF files (DATASETS, src_arf_file,
make_sampled_file) and a fabricated open-ephys recording tree (build_tree).
The latter is shared because the pipeline tests import a tree through
arfx-oephys and then hand the result to collect, splitter and select -- which
is the archival workflow the package exists to support.

DATASETS covers the dataset shapes arfx has to handle -- 1-D and 2-D sampled
data, a spike train on a sample timebase, an empty dataset, and a marked point
process -- so that a test which walks every dataset in an entry exercises all
of the branches in core.dataset_properties and the arf.is_* predicates.

Modules that need the raw specs import DATASETS directly; everything else goes
through the fixtures.
"""

import datetime
import json
import time

import arf
import numpy as np
import pytest
from numpy.random import randint, randn

from arfx import io, oephys

tstamp = time.mktime(time.localtime())

DATASETS = [
    dict(
        name="acoustic",
        data=randn(100000),
        sampling_rate=20000,
        datatype=arf.DataTypes.ACOUSTIC,
        maxshape=(None,),
        microphone="DK-1234",
        compression=0,
    ),
    dict(
        name="neural",
        data=(randn(100000) * 2**16).astype("h"),
        sampling_rate=20000,
        datatype=arf.DataTypes.EXTRAC_HP,
        compression=9,
    ),
    dict(
        name="multichannel",
        data=randn(10000, 2),
        sampling_rate=20000,
        datatype=arf.DataTypes.ACOUSTIC,
    ),
    dict(
        name="spikes",
        data=randint(0, 100000, 100),
        datatype=arf.DataTypes.SPIKET,
        units="samples",
        sampling_rate=20000,  # required
    ),
    dict(
        name="empty-spikes",
        data=np.array([], dtype="f"),
        datatype=arf.DataTypes.SPIKET,
        method="broken",
        maxshape=(None,),
        units="s",
    ),
    dict(
        name="events",
        data=np.rec.fromrecords(
            [(1.0, 1, b"stimulus"), (5.0, 0, b"stimulus")],
            names=("start", "state", "name"),
        ),  # 'start' required
        datatype=arf.DataTypes.EVENT,
        units=(b"s", b"", b""),
    ),  # only bytes supported by h5py
]


@pytest.fixture
def datasets():
    return DATASETS


@pytest.fixture
def src_arf_file(tmp_path):
    path = tmp_path / "input.arf"
    with arf.open_file(path, "w") as fp:
        entry = arf.create_entry(fp, "entry", tstamp)
        for dset in DATASETS:
            _ = arf.create_dataset(entry, **dset)
    return path


@pytest.fixture
def src_wav_files(tmp_path):
    test_dsets = DATASETS[:3]
    test_files = []
    for dset in test_dsets:
        src_file = (tmp_path / dset["name"]).with_suffix(".wav")
        src_data = dset["data"]
        nchannels = src_data.shape[1] if src_data.ndim > 1 else 1
        with io.open(
            src_file,
            mode="w",
            nchannels=nchannels,
            dtype=src_data.dtype,
            sampling_rate=dset["sampling_rate"],
        ) as fp:
            fp.write(dset["data"])
        test_files.append(src_file)
    return test_files


# ----------------------------------------------------- multi-entry sampled data

SAMPLING_RATE = 1000
ENTRY_SAMPLES = 5000
CHANNELS = ("chan_a", "chan_b")


def make_sampled_file(path, nentries=3, samples=ENTRY_SAMPLES, start=tstamp, **attrs):
    """Build an arf file of sampled-only entries with identical channel layout.

    collect.check_entry_consistency requires that every entry expose the same
    channels with the same properties, so this is the shape most of the
    collect and splitter tests need. Entry N starts at `start + N * duration`
    so timestamps are strictly increasing and splitter's sort is deterministic.
    """
    duration = samples / SAMPLING_RATE
    with arf.open_file(path, "w") as fp:
        for i in range(nentries):
            entry = arf.create_entry(fp, f"entry_{i:03}", start + i * duration, **attrs)
            for j, channel in enumerate(CHANNELS):
                # a per-channel ramp offset by channel index, so a test can tell
                # the channels apart after they are interleaved
                data = np.arange(samples, dtype="h") + j * samples
                arf.create_dataset(
                    entry,
                    channel,
                    data,
                    sampling_rate=SAMPLING_RATE,
                    datatype=arf.DataTypes.EXTRAC_HP,
                )
    return path


@pytest.fixture
def sampled_arf_file(tmp_path):
    return make_sampled_file(tmp_path / "sampled.arf")


# ================================ open-ephys ================================

TIMESTAMP_DIR = "P397_2026-06-17_11-08-34_arc6-main"
EXPECTED_TIME = datetime.datetime(2026, 6, 17, 11, 8, 34)
# The wall-clock time GUI >= 0.6 writes into sync_messages.txt, which is where
# the entry timestamp now comes from. It agrees with the directory name to the
# second and carries milliseconds beyond it; both come from a real recording.
SOFTWARE_TIME_MS = 1781708914429
EXPECTED_SOFTWARE_TIME = datetime.datetime.fromtimestamp(SOFTWARE_TIME_MS / 1000)
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


def sync_messages_text(gui_version, software_time_ms=SOFTWARE_TIME_MS):
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
        f"{software_time_ms}\n"
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
    session=TIMESTAMP_DIR,
    experiment=1,
    recording=1,
    software_time_ms=SOFTWARE_TIME_MS,
):
    """Fabricate an open-ephys binary-format recording directory.

    Returns the top-level path, which is what the script takes on the command
    line. The timestamp is parsed out of that directory's name, so it must keep
    the YYYY-MM-DD_HH-MM-SS form.

    `session` names that directory. Entry names are derived from it, so a test
    that puts two recordings in one archive has to give them distinct sessions,
    exactly as two real recordings would have distinct timestamps.

    `experiment` and `recording` place this recording within the session. Newer
    versions of the GUI write several of each into one session directory, so
    calling this repeatedly with the same `root` and `session` builds the tree
    that arfx-oephys has to turn into several entries. Give each one its own
    `software_time_ms`; that is where its timestamp comes from.
    """
    p = PROFILES[gui_version]
    rate = p["sampling_rate"]
    base = (
        root
        / session
        / f"Record Node {p['record_node']}"
        / f"experiment{experiment}"
        / f"recording{recording}"
    )
    base.mkdir(parents=True, exist_ok=True)
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
        (base / "sync_messages.txt").write_text(
            sync_messages_text(gui_version, software_time_ms)
        )

    return root / session


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
