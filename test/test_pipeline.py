# -*- mode: python -*-
"""End-to-end tests across the scripts, following the archival workflow.

An open-ephys recording is imported with arfx-oephys, and the resulting ARF
file is then handed to collect or select. The guarantee that matters is the
first one below: a recording that has been archived to ARF must come back out
as a continuous.dat that a spike sorter will accept, because that is the whole
reason for keeping the archive. Everything else here supports it.

arfx-split is deliberately absent. It is used to chunk long acoustic
recordings into hours or days, not to divide an ephys recording -- an ephys
archive goes straight back to collect for a re-sort -- so its tests live in
test_splitter.py rather than being composed here.

The channel count is deliberately 12. arfx-collect-sampled interleaves channels
in h5py iteration order, and at ten or more channels creation order
(CH1, CH2, ... CH10) and alphabetical order (CH1, CH10, CH11, CH2) diverge --
so a smaller fixture cannot tell a correct implementation from one that would
silently transpose most of a real 136-channel recording.
"""

import json

import arf
import h5py
import numpy as np
import pytest
from conftest import NEW, PROFILES, build_tree, only_entry, recording_dir, run_oephys

from arfx import collect, select

NCHANNELS = 12
NSAMPLES = 600
RATE = PROFILES[NEW]["sampling_rate"]  # 30000.0
DURATION = NSAMPLES / RATE  # 0.02 s


def pipeline_data(nchannels=NCHANNELS, nsamples=NSAMPLES):
    """Interleaved int16 with a distinguishable signed ramp per channel.

    Channel j is offset by j*1000 and centred so that the low channels are
    negative, which catches both a transposition and a sign error.
    """
    data = np.empty((nsamples, nchannels), dtype="int16")
    for j in range(nchannels):
        data[:, j] = (j - nchannels // 2) * 1000 + np.arange(nsamples)
    return data


def continuous_dat(tree, gui_version):
    folder = PROFILES[gui_version]["continuous_folder"].strip("/")
    return recording_dir(tree, gui_version) / "continuous" / folder / "continuous.dat"


def channel_names(tree, gui_version):
    """The channel order structure.oebin declares, which is the order in the .dat."""
    rec = recording_dir(tree, gui_version)
    structure = json.loads((rec / "structure.oebin").read_text())
    return [ch["channel_name"] for ch in structure["continuous"][0]["channels"]]


@pytest.fixture
def tree(tmp_path_factory, gui_version):
    root = tmp_path_factory.mktemp("pipeline")
    return build_tree(root, gui_version, nchannels=NCHANNELS, data=pipeline_data())


@pytest.fixture
def archive(tmp_path, tree):
    """The recording, imported to ARF the way it would be for archival."""
    return run_oephys(tmp_path / "archive.arf", tree)


def read_dat(path, nchannels=NCHANNELS):
    return np.fromfile(path, dtype="int16").reshape(-1, nchannels)


# --------------------------------------------------------- the archival guarantee


def test_continuous_dat_round_trips_byte_for_byte(tmp_path, tree, archive, gui_version):
    # import to ARF, then unpack back to raw binary. The bytes must be
    # identical: this is what makes the ARF file a substitute for the original
    # recording when spike sorting has to be re-run.
    recovered = tmp_path / "recovered.dat"
    collect.collect_sampled_script([str(archive), str(recovered)])
    assert recovered.read_bytes() == continuous_dat(tree, gui_version).read_bytes()


def test_round_trip_preserves_every_sample(tmp_path, tree, archive, gui_version):
    # the same check by value rather than by bytes, so a failure says which
    # channel and sample went wrong instead of just "files differ"
    recovered = tmp_path / "recovered.dat"
    collect.collect_sampled_script([str(archive), str(recovered)])
    original = read_dat(continuous_dat(tree, gui_version))
    assert np.array_equal(read_dat(recovered), original)


def test_channel_order_is_not_alphabetical(tmp_path, tree, archive, gui_version):
    # CH10 must land in column 9, where structure.oebin puts it, not column 1
    # where a sort by name would. Collect relies on h5py returning the datasets
    # in creation order, which holds because arf.create_entry tracks it.
    names = channel_names(tree, gui_version)
    assert names[1] == "CH2" and names[9] == "CH10"  # fixture sanity
    recovered = tmp_path / "recovered.dat"
    collect.collect_sampled_script([str(archive), str(recovered)])
    data = read_dat(recovered)
    expected = pipeline_data()
    for column in (1, 9, 10, 11):
        assert np.array_equal(data[:, column], expected[:, column]), (
            f"column {column} ({names[column]}) does not match"
        )


def test_archive_holds_channels_in_declared_order(archive, tree, gui_version):
    with arf.open_file(archive, "r") as fp:
        entry = only_entry(fp)
        sampled = [
            n for n in arf.keys_by_creation(entry) if arf.is_time_series(entry[n])
        ]
        assert sampled == channel_names(tree, gui_version)


# ------------------------------------------------------------------ import shape


def test_import_preserves_dtype_and_rate(archive):
    with arf.open_file(archive, "r") as fp:
        entry = only_entry(fp)
        for name in ("CH1", "CH10"):
            dset = entry[name]
            assert dset.dtype == np.dtype("int16")
            assert dset.attrs["sampling_rate"] == RATE
            assert dset.shape == (NSAMPLES,)


def test_import_carries_event_streams(archive):
    with arf.open_file(archive, "r") as fp:
        entry = only_entry(fp)
        events = [n for n in entry if arf.is_marked_pointproc(entry[n])]
        assert len(events) == 1
        dset = entry[events[0]]
        assert "start" in dset.dtype.names


# ------------------------------------------------------------- multi-recording


def test_session_with_several_recordings_round_trips(tmp_path, tmp_path_factory):
    # A session can hold more than one recording, and they are rarely the same
    # length. Both land in one archive, and both have to come back out.
    #
    # This is where collect used to fail: it compared the whole channel property
    # dict across entries, and that included chunksize, which h5py derives from
    # each dataset's own length. Two entries with identical channels read as
    # having different ones purely because one was shorter.
    archive = tmp_path / "session.arf"
    originals = []
    for name, nsamples, session in (
        ("r1", 600, "P397_2026-06-17_11-08-34_arc6-main"),
        ("r2", 400, "P397_2026-06-17_12-15-00_arc6-main"),
    ):
        data = pipeline_data(nsamples=nsamples)
        tree = build_tree(
            tmp_path_factory.mktemp(name),
            NEW,
            nchannels=NCHANNELS,
            data=data,
            session=session,
        )
        run_oephys(archive, tree)
        originals.append(data)

    with arf.open_file(archive, "r") as fp:
        entries = [n for n in arf.keys_by_creation(fp) if arf.is_entry(fp[n])]
        assert [fp[n]["CH1"].shape[0] for n in entries] == [600, 400]

    recovered = tmp_path / "recovered.dat"
    assert collect.collect_sampled_script([str(archive), str(recovered)]) != 1
    assert np.array_equal(read_dat(recovered), np.concatenate(originals))


def test_collect_accepts_entries_of_different_lengths(tmp_path):
    # the same tolerance at the unit level, without the open-ephys machinery
    path = tmp_path / "ragged.arf"
    with arf.open_file(path, "w") as fp:
        for i, count in enumerate((5000, 3000, 4000)):
            entry = arf.create_entry(fp, f"entry_{i}", 1000 + i)
            arf.create_dataset(
                entry, "ch", np.full(count, i, dtype="h"), sampling_rate=1000
            )
    out = tmp_path / "out.dat"
    assert collect.collect_sampled_script([str(path), str(out)]) != 1
    data = np.fromfile(out, dtype="int16")
    assert data.tolist() == [0] * 5000 + [1] * 3000 + [2] * 4000


# ------------------------------------------------------------------ select


def test_select_from_imported_recording(tmp_path, archive):
    segments = tmp_path / "segments.json"
    begin, end = 0.002, 0.010
    segments.write_text(json.dumps({"entry": 0, "begin": begin, "end": end}) + "\n")
    out = tmp_path / "selected.arf"
    select.main(["-s", str(segments), str(archive), str(out)])

    expected = pipeline_data()
    first, last = int(begin * RATE), int(end * RATE)
    with arf.open_file(out, "r") as fp:
        entry = fp["entry_00000"]
        # every sampled channel plus the message-event dataset
        assert len(entry) == NCHANNELS + 1
        for name, index in (("CH1", 0), ("CH2", 1), ("CH10", 9)):
            dset = entry[name]
            assert dset.shape == (last - first,)
            assert np.array_equal(dset[:], expected[first:last, index])


# ------------------------------------------------- archival edge cases (#48)


def rewrite_structure(tree, gui_version, edit):
    path = recording_dir(tree, gui_version) / "structure.oebin"
    structure = json.loads(path.read_text())
    edit(structure["continuous"][0]["channels"])
    path.write_text(json.dumps(structure))


def collect_to(tmp_path, archive, *extra, name="out.dat"):
    out = tmp_path / name
    assert collect.collect_sampled_script([str(archive), str(out), *extra]) != 1
    return out


def test_headstage_and_adc_channels(tmp_path, gui_version):
    # ADC channels follow the headstage channels, with their own bit_volts and
    # units. Collect writes raw integers and must not apply either.
    nhead, nadc = 12, 3
    n = nhead + nadc
    tree = build_tree(tmp_path / "src", gui_version, nchannels=n, data=pipeline_data(n))

    def edit(channels):
        for i, ch in enumerate(channels[nhead:], start=1):
            ch.update(channel_name=f"ADC{i}", bit_volts=0.000050354, units="V")

    rewrite_structure(tree, gui_version, edit)
    archive = run_oephys(tmp_path / "a.arf", tree)
    out = collect_to(tmp_path, archive)
    assert out.read_bytes() == continuous_dat(tree, gui_version).read_bytes()
    with arf.open_file(archive, "r") as fp:
        entry = only_entry(fp)
        names = [k for k in arf.keys_by_creation(entry) if arf.is_time_series(entry[k])]
    assert names[nhead:] == ["ADC1", "ADC2", "ADC3"]


def test_counter_reset_does_not_change_bytes(tmp_path, gui_version):
    # the offset is derived from the final contiguous run, and may be negative;
    # none of that may touch the samples
    data = pipeline_data()
    numbers = np.concatenate([np.arange(100, 100 + 200), np.arange(0, NSAMPLES - 200)])
    tree = build_tree(
        tmp_path / "src",
        gui_version,
        nchannels=NCHANNELS,
        data=data,
        sample_numbers=numbers,
    )
    archive = run_oephys(tmp_path / "a.arf", tree)
    out = collect_to(tmp_path, archive)
    assert out.read_bytes() == continuous_dat(tree, gui_version).read_bytes()


def test_multiple_chunks_with_partial_final_chunk(tmp_path, gui_version):
    nsamples = 200_123
    tree = build_tree(
        tmp_path / "src",
        gui_version,
        nchannels=NCHANNELS,
        data=pipeline_data(NCHANNELS, nsamples),
        sample_numbers=np.arange(nsamples),
    )
    archive = run_oephys(tmp_path / "a.arf", tree)
    with arf.open_file(archive, "r") as fp:
        dset = only_entry(fp)["CH1"]
        chunk = dset.chunks[0]
    assert nsamples > chunk and nsamples % chunk != 0  # fixture sanity
    out = collect_to(tmp_path, archive)
    assert out.read_bytes() == continuous_dat(tree, gui_version).read_bytes()


def test_single_recording_from_multi_recording_session(tmp_path):
    root = tmp_path / "src"
    datas = []
    for rec, nsamples in ((1, 600), (2, 400)):
        data = pipeline_data(nsamples=nsamples) + rec
        datas.append(data)
        build_tree(
            root,
            NEW,
            nchannels=NCHANNELS,
            data=data,
            recording=rec,
            software_time_ms=1781708914429 + rec * 60_000,
        )
    archive = run_oephys(
        tmp_path / "a.arf", root / "P397_2026-06-17_11-08-34_arc6-main"
    )
    with arf.open_file(archive, "r") as fp:
        entries = [n for n in arf.keys_by_creation(fp) if arf.is_entry(fp[n])]
    assert len(entries) == 2
    for entry, data in zip(entries, datas, strict=True):
        out = collect_to(tmp_path, archive, "-e", entry, name=f"{entry}.dat")
        assert np.array_equal(read_dat(out), data)


def test_channel_option_sets_output_order(tmp_path, tree, archive, caplog):
    import logging

    wanted = ["CH10", "CH2", "CH1"]
    with caplog.at_level(logging.INFO, logger="arfx-collect"):
        out = collect_to(tmp_path, archive, "-c", *wanted)
    data = read_dat(out, 3)
    expected = pipeline_data()
    assert np.array_equal(data, expected[:, [9, 1, 0]])
    logged = [r.getMessage().strip("- ") for r in caplog.records]
    positions = [logged.index(c) for c in wanted]
    assert positions == sorted(positions)


def test_untracked_entries_are_refused(tmp_path, caplog):
    # without creation order h5py iterates alphabetically; CH10 would land
    # before CH2 with nothing to say so
    path = tmp_path / "untracked.arf"
    with h5py.File(path, "w") as fp:
        fp.attrs["arf_version"] = arf.spec_version
        entry = fp.create_group("entry")
        entry.attrs["timestamp"] = np.array([1000, 0], dtype="uint64")
        for name in ("CH1", "CH2", "CH10"):
            dset = entry.create_dataset(
                name, data=np.zeros(10, dtype="int16"), chunks=True
            )
            dset.attrs["sampling_rate"] = 1000.0
            dset.attrs["datatype"] = 1
    out = tmp_path / "out.dat"
    assert collect.collect_sampled_script([str(path), str(out)]) == 1
    assert "creation order" in caplog.text
    # naming the channels says what order was meant, so that is accepted
    assert (
        collect.collect_sampled_script(
            [str(path), str(out), "-c", "CH1", "CH2", "CH10"]
        )
        != 1
    )
