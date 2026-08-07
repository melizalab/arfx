# -*- mode: python -*-
"""Tests for arfx.select (the arfx-select script).

select.py is the heaviest user of the arf API in the package -- select_interval,
keys_by_creation, is_entry, is_marked_pointproc and create_entry all appear here
and nowhere else that is tested -- so it is the module most exposed to an arf
version bump.

Segments are fed through a file rather than stdin: the -s argument is
`type=open`, so a path is all that is needed and the tests do not have to
monkeypatch sys.stdin.

"""

import json

import arf
import numpy as np
import pytest

from arfx import select

SAMPLING_RATE = 1000
NSAMPLES = 10000  # 10 seconds
EVENT_RECORDS = [(500, b"a"), (5500, b"b"), (9500, b"c")]


def add_marked(entry, units):
    return arf.create_dataset(
        entry,
        "events",
        np.rec.fromrecords(EVENT_RECORDS, names=("start", "name")),
        units=units,
        sampling_rate=SAMPLING_RATE,
        datatype=arf.DataTypes.EVENT,
    )


@pytest.fixture
def src_file(tmp_path):
    """Two entries holding sampled data and a simple point process."""
    path = tmp_path / "src.arf"
    with arf.open_file(path, "w") as fp:
        fp.attrs["experimenter"] = "dmeliza"
        # a top-level dataset, which select copies over wholesale
        fp.create_dataset("toplevel", data=np.arange(10))
        for i in range(2):
            entry = arf.create_entry(fp, f"entry_{i:02}", 1000 + i, pen=i)
            arf.create_dataset(
                entry,
                "sampled",
                np.arange(NSAMPLES, dtype="h"),
                sampling_rate=SAMPLING_RATE,
                datatype=arf.DataTypes.EXTRAC_HP,
            )
            arf.create_dataset(
                entry,
                "spikes",
                np.arange(0, NSAMPLES, 500),
                units="samples",
                sampling_rate=SAMPLING_RATE,
                datatype=arf.DataTypes.SPIKET,
            )
    return path


@pytest.fixture
def marked_file(tmp_path):
    """One entry with a spec-conforming marked point process (one unit per field)."""
    path = tmp_path / "marked.arf"
    with arf.open_file(path, "w") as fp:
        entry = arf.create_entry(fp, "entry", 1000)
        add_marked(entry, (b"samples", b""))
    return path


@pytest.fixture
def jrecord_file(tmp_path):
    """A marked dataset with jrecord's non-conforming scalar units attribute."""
    path = tmp_path / "jrecord.arf"
    with arf.open_file(path, "w") as fp:
        entry = arf.create_entry(fp, "entry", 1000)
        dset = add_marked(entry, (b"samples", b""))
        del dset.attrs["units"]
        dset.attrs["units"] = "samples"
    return path


def write_segments(path, *segments):
    path.write_text("\n".join(json.dumps(s) for s in segments) + "\n")
    return path


def run_select(tmp_path, src, segments, *extra):
    """Drive the script end to end and return the path it wrote."""
    seg_file = write_segments(tmp_path / "segments.json", *segments)
    tgt = tmp_path / "tgt.arf"
    select.main([*extra, "-s", str(seg_file), str(src), str(tgt)])
    return tgt


# ------------------------------------------------------------ entry addressing


def test_select_entry_by_name(tmp_path, src_file):
    tgt = run_select(
        tmp_path, src_file, [{"entry": "entry_01", "begin": 1.0, "end": 3.0}]
    )
    with arf.open_file(tgt, "r") as fp:
        assert sorted(fp.keys()) == ["entry_00000", "toplevel"]
        assert fp["entry_00000"]["sampled"].shape == (2000,)


def test_select_entry_by_index(tmp_path, src_file):
    # an integer indexes into the creation-ordered list of entries
    tgt = run_select(tmp_path, src_file, [{"entry": 1, "begin": 1.0, "end": 3.0}])
    with arf.open_file(tgt, "r") as fp:
        # entry_01 was created with pen=1; this is how we know which one we got
        assert fp["entry_00000"].attrs["pen"] == 1


def test_select_multiple_segments_are_numbered_sequentially(tmp_path, src_file):
    tgt = run_select(
        tmp_path,
        src_file,
        [
            {"entry": "entry_00", "begin": 0.0, "end": 1.0},
            {"entry": "entry_01", "begin": 1.0, "end": 2.0},
            {"entry": "entry_00", "begin": 2.0, "end": 3.0},
        ],
    )
    with arf.open_file(tgt, "r") as fp:
        names = [n for n in arf.keys_by_creation(fp) if arf.is_entry(fp[n])]
        assert names == ["entry_00000", "entry_00001", "entry_00002"]


# ------------------------------------------------------------------- selection


def test_sampled_data_is_sliced_and_offset(tmp_path, src_file):
    tgt = run_select(
        tmp_path, src_file, [{"entry": "entry_00", "begin": 2.0, "end": 4.0}]
    )
    with arf.open_file(tgt, "r") as fp:
        dset = fp["entry_00000"]["sampled"]
        assert dset.shape == (2000,)
        # the source is a ramp, so the first value identifies the slice
        assert dset[0] == 2000
        assert dset.attrs["offset"] == 2000
        assert dset.attrs["sampling_rate"] == SAMPLING_RATE


def test_point_process_times_are_rebased(tmp_path, src_file):
    tgt = run_select(
        tmp_path, src_file, [{"entry": "entry_00", "begin": 2.0, "end": 4.0}]
    )
    with arf.open_file(tgt, "r") as fp:
        # source spikes are every 500 samples; [2000, 4000) holds 2000/2500/3000/3500
        assert fp["entry_00000"]["spikes"][:].tolist() == [0, 500, 1000, 1500]


def test_entry_attributes_are_carried_over(tmp_path, src_file):
    tgt = run_select(
        tmp_path, src_file, [{"entry": "entry_01", "begin": 1.0, "end": 2.0}]
    )
    with arf.open_file(tgt, "r") as fp:
        entry = fp["entry_00000"]
        assert entry.attrs["pen"] == 1
        # create_entry re-derives the uuid but the source timestamp is preserved
        assert "timestamp" in entry.attrs
        assert "uuid" in entry.attrs


def test_selection_beyond_end_of_data_is_empty(tmp_path, src_file):
    tgt = run_select(
        tmp_path, src_file, [{"entry": "entry_00", "begin": 20.0, "end": 30.0}]
    )
    with arf.open_file(tgt, "r") as fp:
        assert fp["entry_00000"]["sampled"].shape == (0,)


def test_offsets_accumulate(tmp_path):
    # select.py adds the interval offset to whatever offset the source dataset
    # already carried, so selecting out of an already-selected file composes.
    # Built directly rather than by chaining two runs, which keeps the
    # arithmetic being tested visible in one place.
    path = tmp_path / "offset.arf"
    with arf.open_file(path, "w") as fp:
        entry = arf.create_entry(fp, "entry", 1000)
        arf.create_dataset(
            entry,
            "sampled",
            np.arange(NSAMPLES, dtype="h"),
            sampling_rate=SAMPLING_RATE,
            offset=2000,
        )
    tgt = run_select(tmp_path, path, [{"entry": "entry", "begin": 1.0, "end": 2.0}])
    with arf.open_file(tgt, "r") as fp:
        assert fp["entry_00000"]["sampled"].attrs["offset"] == 3000


# -------------------------------------------------------------- channel subset


def test_channels_flag_restricts_output(tmp_path, src_file):
    tgt = run_select(
        tmp_path,
        src_file,
        [{"entry": "entry_00", "begin": 1.0, "end": 2.0}],
        "-c",
        "sampled",
    )
    with arf.open_file(tgt, "r") as fp:
        assert list(fp["entry_00000"].keys()) == ["sampled"]


# ---------------------------------------------------- marked point process data


def test_conforming_marked_units_round_trip(tmp_path, marked_file):
    # Two separate things had to be true for this to work, and neither was:
    # the units attribute had to survive the jrecord repair block (it was
    # popped and only restored in the scalar case), and it had to be handed to
    # create_dataset as a list rather than the ndarray h5py hands back.
    tgt = run_select(
        tmp_path, marked_file, [{"entry": "entry", "begin": 5.0, "end": 6.0}]
    )
    with arf.open_file(tgt, "r") as fp:
        events = fp["entry_00000"]["events"]
        assert list(events.attrs["units"]) == ["samples", ""]
        assert events["start"].tolist() == [500]  # 5500 rebased
        assert events["name"].tolist() == [b"b"]


def test_malformed_units_on_marked_dataset_raises(tmp_path, jrecord_file):
    # CHARACTERIZATION: this is an arf 2.7.3 regression, not an arfx bug, and it
    # breaks the one input select.py's units workaround was written to handle.
    #
    # arf._sample_timebase looks up the unit for the 'start' field with
    # units[names.index("start")]. For jrecord's scalar "samples" that indexes
    # the *string*, yielding "s", so the dataset is not recognized as being on a
    # sample timebase. `begin` is then left as a float, and rebasing the int64
    # start times with `data["start"] -= begin` fails the same-kind cast.
    #
    # Under 2.7.2 the conversion was unconditional whenever sampling_rate was
    # present, so begin was always an int and this path worked. The fix belongs
    # in arf: treat a scalar units attribute on a compound dataset as malformed
    # rather than indexing into it.
    with pytest.raises(TypeError, match="same_kind"):
        run_select(
            tmp_path, jrecord_file, [{"entry": "entry", "begin": 0.0, "end": 1.0}]
        )


def test_preserve_marked_copies_without_selecting(tmp_path, marked_file):
    tgt = run_select(
        tmp_path,
        marked_file,
        [{"entry": "entry", "begin": 5.0, "end": 6.0}],
        "--preserve-marked",
    )
    with arf.open_file(tgt, "r") as fp:
        events = fp["entry_00000"]["events"]
        # all three records survive, with original (un-rebased) times
        assert events["start"].tolist() == [500, 5500, 9500]


# ---------------------------------------------------------- top-level contents


def test_toplevel_datasets_and_attributes_are_copied(tmp_path, src_file):
    tgt = run_select(
        tmp_path, src_file, [{"entry": "entry_00", "begin": 1.0, "end": 2.0}]
    )
    with arf.open_file(tgt, "r") as fp:
        assert fp["toplevel"][:].tolist() == list(range(10))
        assert fp.attrs["experimenter"] == "dmeliza"
        # the target's own version stamp must not be clobbered by the source's
        assert fp.attrs["arf_version"] == arf.spec_version


# ------------------------------------------------------------------ error paths


def test_malformed_json_is_skipped(tmp_path, src_file, caplog):
    seg_file = tmp_path / "segments.json"
    seg_file.write_text(
        "this is not json\n"
        + json.dumps({"entry": "entry_00", "begin": 1.0, "end": 2.0})
        + "\n"
    )
    tgt = tmp_path / "tgt.arf"
    select.main(["-s", str(seg_file), str(src_file), str(tgt)])
    with arf.open_file(tgt, "r") as fp:
        # the good line still produced an entry
        assert "entry_00000" in fp
    assert "invalid json" in caplog.text


def test_missing_entry_key_is_reported(tmp_path, src_file, caplog):
    tgt = run_select(tmp_path, src_file, [{"begin": 1.0, "end": 2.0}])
    with arf.open_file(tgt, "r") as fp:
        assert not [n for n in fp if arf.is_entry(fp[n])]
    assert "entry" in caplog.text


def test_unknown_entry_name_is_reported(tmp_path, src_file, caplog):
    run_select(
        tmp_path, src_file, [{"entry": "no_such_entry", "begin": 0.0, "end": 1.0}]
    )
    assert "no_such_entry" in caplog.text


# ---------------------------------------------------------------------- dry run


def test_dry_run_writes_nothing(tmp_path, src_file):
    seg_file = write_segments(
        tmp_path / "segments.json", {"entry": "entry_00", "begin": 1.0, "end": 2.0}
    )
    tgt = tmp_path / "tgt.arf"
    select.main(["--dry-run", "-s", str(seg_file), str(src_file), str(tgt)])
    assert not tgt.exists()
