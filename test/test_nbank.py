"""Tests for the neurobank metadata reader (arfx.nbank)"""

import json

import arf
import h5py
import numpy as np
import pytest
from conftest import TIMESTAMP_DIR, build_tree, make_sampled_file, run_oephys

from arfx import __version__, nbank

OTHER_SESSION = "P397_2026-06-17_15-30-00_arc6-main"

PROCESSED_BY = {"processed_by": "org.meliza.arfx/arfx-oephys " + __version__}


def read(path):
    return nbank.read_metadata(path)


def test_registered_as_entry_point():
    from importlib.metadata import entry_points

    eps = entry_points(group="nbank.metadata_readers")
    for dtype in ("extracellular-arf", "eeg-arf"):
        assert eps[dtype].load() is nbank.read_metadata


def test_reads_the_attributes_given_with_k(tmp_path, tree):
    tgt = run_oephys(
        tmp_path / "out.arf",
        tree,
        extra=[
            "-k",
            "animal=0d1a2b3c-0000-4000-8000-000000000001",
            "-k",
            "experimenter=dmeliza",
            "-k",
            "recording=r1",
            "-k",
            "pen=1",
        ],
    )
    md = read(tgt)
    assert md == {
        "animal": "0d1a2b3c-0000-4000-8000-000000000001",
        "experimenter": "dmeliza",
        "recording": "r1",
        "pen": 1,
        "processed_by": "org.meliza.arfx/arfx-oephys " + __version__,
    }
    json.dumps(md)


def test_reads_experimenter_and_animal_options(tmp_path, tree):
    animal = "b1360fc4-34b8-41cb-a7c7-ec5d774238e9"
    tgt = run_oephys(
        tmp_path / "out.arf",
        tree,
        extra=["--experimenter", "tbyron", "--animal", animal],
    )
    assert read(tgt) == {"experimenter": "tbyron", "animal": animal} | PROCESSED_BY


def test_only_user_attributes_are_returned(tmp_path, tree):
    # timestamp, uuid, entry_creator and the open-ephys structure stay out
    tgt = run_oephys(tmp_path / "out.arf", tree)
    assert read(tgt) == PROCESSED_BY


def test_file_is_unchanged(tmp_path, tree):
    tgt = run_oephys(tmp_path / "out.arf", tree, extra=["-k", "pen=1"])
    before = tgt.read_bytes()
    read(tgt)
    assert tgt.read_bytes() == before


def test_entries_that_agree(tmp_path, tmp_path_factory, gui_version):
    a = build_tree(tmp_path_factory.mktemp("a"), gui_version)
    b = build_tree(tmp_path_factory.mktemp("b"), gui_version, session=OTHER_SESSION)
    tgt = tmp_path / "out.arf"
    run_oephys(tgt, a, extra=["-k", "animal=x"])
    run_oephys(tgt, b, extra=["-k", "animal=x"])
    assert read(tgt) == {"animal": "x"} | PROCESSED_BY


def two_entry_file(tmp_path, tmp_path_factory, gui_version, first, second):
    tgt = tmp_path / "out.arf"
    for name, session, extra in (
        ("a", TIMESTAMP_DIR, first),
        ("b", OTHER_SESSION, second),
    ):
        tree = build_tree(tmp_path_factory.mktemp(name), gui_version, session=session)
        run_oephys(tgt, tree, extra=extra)
    return tgt


def test_varying_attribute_is_left_out(tmp_path, tmp_path_factory, gui_version):
    tgt = two_entry_file(
        tmp_path,
        tmp_path_factory,
        gui_version,
        ["-k", "animal=x", "-k", "recording=1"],
        ["-k", "animal=x", "-k", "recording=2"],
    )
    assert read(tgt) == {"animal": "x"} | PROCESSED_BY


def test_attribute_missing_from_some_entries_is_left_out(
    tmp_path, tmp_path_factory, gui_version
):
    tgt = two_entry_file(
        tmp_path,
        tmp_path_factory,
        gui_version,
        ["-k", "animal=x", "-k", "recording=1"],
        ["-k", "animal=x"],
    )
    assert read(tgt) == {"animal": "x"} | PROCESSED_BY


@pytest.mark.parametrize("key", ["animal", "experimenter"])
def test_entries_disagreeing_about_identity_are_refused(
    tmp_path, tmp_path_factory, gui_version, key
):
    tgt = two_entry_file(
        tmp_path,
        tmp_path_factory,
        gui_version,
        ["-k", f"{key}=x"],
        ["-k", f"{key}=y"],
    )
    with pytest.raises(ValueError, match=key):
        read(tgt)


def test_identity_missing_from_some_entries_is_refused(
    tmp_path, tmp_path_factory, gui_version
):
    tgt = two_entry_file(
        tmp_path, tmp_path_factory, gui_version, ["-k", "animal=x"], []
    )
    with pytest.raises(ValueError, match="animal"):
        read(tgt)


def test_legacy_file_uses_known_keys_and_current_names(tmp_path, tree):
    tgt = run_oephys(
        tmp_path / "out.arf",
        tree,
        extra=["-k", "bird=C194", "-k", "experiment=ep1", "-k", "pen=1"],
    )
    with h5py.File(tgt, "a") as fp:
        for name in fp:
            del fp[name].attrs[nbank.USER_ATTRIBUTES]
    assert read(tgt) == {"animal": "C194", "protocol": "ep1"} | PROCESSED_BY


def test_no_k_is_not_legacy(tmp_path, tree):
    tgt = run_oephys(tmp_path / "out.arf", tree)
    with arf.open_file(tgt, "r") as fp:
        (name,) = [n for n in fp if arf.is_entry(fp[n])]
        assert len(fp[name].attrs[nbank.USER_ATTRIBUTES]) == 0


def test_numpy_and_bytes_become_json(tmp_path, tree):
    tgt = run_oephys(tmp_path / "out.arf", tree)
    with h5py.File(tgt, "a") as fp:
        (name,) = [n for n in fp if arf.is_entry(fp[n])]
        e = fp[name]
        e.attrs["protocol"] = np.float32(0.5)
        e.attrs["recording"] = np.int64(3)
        e.attrs["presenter"] = np.array([1, 2], dtype="i2")
        e.attrs["animal"] = np.bytes_(b"abc")
        e.attrs["experimenter"] = np.array([1.0, np.nan])
        e.attrs[nbank.USER_ATTRIBUTES] = np.array(
            [
                "protocol",
                "presenter",
                "experimenter",
                "recording",
                "animal",
            ],
            dtype=h5py.string_dtype(),
        )
    md = read(tgt)
    assert md == {
        "protocol": 0.5,
        "recording": 3,
        "presenter": [1, 2],
        "animal": "abc",
        "processed_by": "org.meliza.arfx/arfx-oephys " + __version__,
    }
    assert type(md["recording"]) is int
    json.dumps(md, allow_nan=False)


@pytest.mark.parametrize(
    ("old", "new"), [("bird", "animal"), ("experiment", "protocol")]
)
def test_old_and_new_names_disagreeing_are_refused(tmp_path, tree, old, new):
    tgt = run_oephys(
        tmp_path / "out.arf", tree, extra=["-k", f"{old}=x", "-k", f"{new}=y"]
    )
    with pytest.raises(ValueError, match=old):
        read(tgt)


def test_old_and_new_names_agreeing(tmp_path, tree):
    tgt = run_oephys(
        tmp_path / "out.arf", tree, extra=["-k", "bird=x", "-k", "animal=x"]
    )
    assert read(tgt) == {"animal": "x"} | PROCESSED_BY


@pytest.mark.parametrize("value", ["1", "[1, 2]"])
def test_processed_by_not_a_string_is_refused(tmp_path, tree, value):
    tgt = run_oephys(tmp_path / "out.arf", tree, extra=["-k", f"processed_by={value}"])
    with pytest.raises(ValueError, match="processed_by"):
        read(tgt)


@pytest.mark.parametrize("rate", ["fast", np.array([1.0, 2.0])])
def test_sampling_rate_not_a_number(tmp_path, rate):
    p = tmp_path / "x.arf"
    make_sampled_file(p, nentries=1)
    with h5py.File(p, "a") as fp:
        for entry in fp.values():
            for dset in entry.values():
                if "sampling_rate" in dset.attrs:
                    dset.attrs["sampling_rate"] = rate
    with pytest.raises(ValueError, match="sampling rate"):
        read(p)


def test_not_hdf5(tmp_path):
    p = tmp_path / "x.arf"
    p.write_text("hello")
    with pytest.raises(ValueError, match="HDF5"):
        read(p)


def test_hdf5_without_arf_root(tmp_path):
    p = tmp_path / "x.arf"
    with h5py.File(p, "w") as fp:
        fp.create_group("e")
    with pytest.raises(ValueError, match="ARF"):
        read(p)


def test_no_entries(tmp_path):
    p = tmp_path / "x.arf"
    with arf.open_file(p, "w"):
        pass
    with pytest.raises(ValueError, match="no entries"):
        read(p)


def test_entry_without_sampled_data(tmp_path):
    p = tmp_path / "x.arf"
    with arf.open_file(p, "w") as fp:
        arf.create_entry(fp, "e", 1.0)
    with pytest.raises(ValueError, match="sampling rate"):
        read(p)


def test_entry_with_sampled_data_but_no_user_list_is_legacy(tmp_path):
    p = tmp_path / "x.arf"
    make_sampled_file(p, nentries=2, bird="B1")
    assert read(p) == {"animal": "B1"}  # no entry_creator


def test_processed_by_lists_each_version_once(tmp_path, tmp_path_factory, gui_version):
    tgt = two_entry_file(tmp_path, tmp_path_factory, gui_version, [], [])
    with h5py.File(tgt, "a") as fp:
        names = sorted(fp)
        fp[names[0]].attrs["entry_creator"] = "org.meliza.arfx/arfx-oephys 3.0.0"
        fp[names[1]].attrs["entry_creator"] = "org.meliza.arfx/arfx-oephys 3.0.2"
    assert read(tgt) == {
        "processed_by": "org.meliza.arfx/arfx-oephys 3.0.0; "
        "org.meliza.arfx/arfx-oephys 3.0.2"
    }


def test_processed_by_given_with_k_wins_over_entry_creator(tmp_path, tree):
    tgt = run_oephys(tmp_path / "out.arf", tree, extra=["-k", "processed_by=someone"])
    assert read(tgt) == {"processed_by": "someone"}
