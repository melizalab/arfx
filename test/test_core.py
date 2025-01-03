# -*- coding: utf-8 -*-
# -*- mode: python -*-
import sys
import time
from pathlib import Path

import arf
import numpy as np
import pytest
from numpy.random import randint, randn

from arfx import core, io

entry_base = "entry_%03d"
tstamp = time.mktime(time.localtime())
entry_attributes = {
    "intattr": 1,
    "vecattr": [1, 2, 3],
    "arrattr": randn(5),
    "strattr": "an attribute",
}
datasets = [
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
def src_arf_file(tmp_path):
    path = tmp_path / "input.arf"
    with arf.open_file(path, "w") as fp:
        entry = arf.create_entry(fp, "entry", tstamp)
        for dset in datasets:
            _ = arf.create_dataset(entry, **dset)
    return path


@pytest.fixture
def src_wav_files(tmp_path):
    test_dsets = datasets[:3]
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


def test_add_entries(src_wav_files, tmp_path):
    tgt_file = tmp_path / "output.arf"
    core.add_entries(tgt_file, src_wav_files)
    with arf.open_file(tgt_file, "r") as fp:
        assert len(fp) == 3
        for dset, entry in zip(datasets, fp.values()):
            assert Path(entry.name).name == dset["name"]
            d = entry["pcm"]  # data always stored as pcm
            assert d.attrs["sampling_rate"] == dset["sampling_rate"]
            assert d.shape == dset["data"].shape
            assert np.all(d[:] == dset["data"])


def test_add_entries_with_metadata(src_wav_files, tmp_path):
    tgt_file = tmp_path / "output.arf"
    core.add_entries(
        tgt_file,
        src_wav_files,
        datatype=arf.DataTypes.ACOUSTIC,
        attrs={"my_attr": "test_value"},
    )
    with arf.open_file(tgt_file, "r") as fp:
        assert len(fp) == 3
        for dset, entry in zip(datasets, fp.values()):
            assert Path(entry.name).name == dset["name"]
            assert entry.attrs["my_attr"] == "test_value"
            d = entry["pcm"]  # data always stored as pcm
            assert d.attrs["datatype"] == arf.DataTypes.ACOUSTIC


def test_add_entries_with_template(src_wav_files, tmp_path):
    tgt_file = tmp_path / "output.arf"
    core.add_entries(tgt_file, src_wav_files, template="entry")
    with arf.open_file(tgt_file, "r") as fp:
        assert len(fp) == 3
        for dset, entry in zip(datasets, fp.values()):
            d = entry["pcm"]  # data always stored as pcm
            assert d.attrs["sampling_rate"] == dset["sampling_rate"]
            assert d.shape == dset["data"].shape
            assert np.all(d[:] == dset["data"])


def test_script_add_entries(src_wav_files, tmp_path):
    tgt_file = tmp_path / "output.arf"
    src_wav_files = [str(path) for path in src_wav_files]
    argv = [
        "-cvf",
        str(tgt_file),
        "-T",
        "ACOUSTIC",
        "-k",
        "this=that",
        "-z 9",
        *src_wav_files,
    ]
    core.arfx(argv)
    with arf.open_file(tgt_file, "r") as fp:
        assert len(fp) == 3
        for dset, entry in zip(datasets, fp.values()):
            assert Path(entry.name).name == dset["name"]
            d = entry["pcm"]  # data always stored as pcm
            assert d.attrs["sampling_rate"] == dset["sampling_rate"]
            assert d.shape == dset["data"].shape
            assert np.all(d[:] == dset["data"])


def test_extract_entries(src_arf_file, tmp_path):
    core.extract_entries(src_arf_file, directory=tmp_path)
    # only the sampled data can be extracted
    for dset in datasets[:3]:
        tgt_file = tmp_path / f"entry_{dset['name']}.wav"
        assert tgt_file.exists()
        with io.open(tgt_file, "r") as fp:
            assert fp.sampling_rate == dset["sampling_rate"]
            data = fp.read()
            assert data.shape == dset["data"].shape
            assert np.all(data == dset["data"])


def test_extract_entries_with_template(src_arf_file, tmp_path):
    core.extract_entries(
        src_arf_file, directory=tmp_path, template="entry_{index:04}_{channel}.wav"
    )
    # only the sampled data can be extracted
    for dset in datasets[:3]:
        tgt_file = tmp_path / f"entry_0000_{dset['name']}.wav"
        assert tgt_file.exists()
        with io.open(tgt_file, "r") as fp:
            assert fp.sampling_rate == dset["sampling_rate"]
            data = fp.read()
            assert data.shape == dset["data"].shape
            assert np.all(data == dset["data"])


def test_extract_entry(src_arf_file, tmp_path):
    core.extract_entries(src_arf_file, ["entry"], directory=tmp_path)
    # only the sampled data can be extracted
    for dset in datasets[:3]:
        tgt_file = tmp_path / f"entry_{dset['name']}.wav"
        assert tgt_file.exists()
        with io.open(tgt_file, "r") as fp:
            assert fp.sampling_rate == dset["sampling_rate"]
            data = fp.read()
            assert data.shape == dset["data"].shape
            assert np.all(data == dset["data"])


def test_extract_nonexistent_entry(src_arf_file, tmp_path):
    core.extract_entries(src_arf_file, ["no_such_entry"], directory=tmp_path)
    for dset in datasets[:3]:
        tgt_file = tmp_path / f"entry_{dset['name']}.wav"
        assert not tgt_file.exists()


def test_script_extract_entries(src_arf_file, tmp_path):
    argv = ["-xvf", str(src_arf_file), "--directory", str(tmp_path)]
    core.arfx(argv)
    for dset in datasets[:3]:
        tgt_file = tmp_path / f"entry_{dset['name']}.wav"
        assert tgt_file.exists()
        with io.open(tgt_file, "r") as fp:
            assert fp.sampling_rate == dset["sampling_rate"]
            data = fp.read()
            assert data.shape == dset["data"].shape
            assert np.all(data == dset["data"])


def test_delete_entry(src_arf_file):
    core.delete_entries(src_arf_file, ["entry"])
    with arf.open_file(src_arf_file, "r") as fp:
        assert len(fp) == 0


def test_delete_nonexistent_entry(src_arf_file):
    core.delete_entries(src_arf_file, ["no_such_entry"])
    with arf.open_file(src_arf_file, "r") as fp:
        assert "entry" in fp


def test_update_all_entries(src_arf_file):
    core.update_entries(src_arf_file, None, my_attr="test_value")
    with arf.open_file(src_arf_file, "r") as fp:
        assert fp["entry"].attrs["my_attr"] == "test_value"


def test_update_entry(src_arf_file):
    core.update_entries(src_arf_file, ["entry"], my_attr="test_value")
    with arf.open_file(src_arf_file, "r") as fp:
        assert fp["entry"].attrs["my_attr"] == "test_value"


def test_update_nonexistent_entry(src_arf_file):
    core.update_entries(src_arf_file, ["no_such_entry"], my_attr="test_value")
    with arf.open_file(src_arf_file, "r") as fp:
        assert "my_attr" not in fp["entry"].attrs


def test_copy_file(src_arf_file, tmp_path):
    tgt_file = tmp_path / "output.arf"
    core.copy_entries(tgt_file, [src_arf_file])

    with arf.open_file(tgt_file, "r") as fp:
        entry = fp["/entry"]
        assert len(entry) == len(datasets)
        assert set(entry.keys()) == set(dset["name"] for dset in datasets)
        # this will fail if iteration is not in order of creation
        for dset, d in zip(datasets, entry.values()):
            assert d.shape == dset["data"].shape
            assert not arf.is_entry(d)


def test_copy_files(src_arf_file, tmp_path):
    tgt_file = tmp_path / "output.arf"
    with pytest.raises(RuntimeError):
        # names will collide and produce error after copying one entry
        core.copy_entries(tgt_file, [src_arf_file, src_arf_file])

    core.copy_entries(tgt_file, [src_arf_file, src_arf_file], entry_base="new_entry")
    fp = arf.open_file(tgt_file, "r")
    print(fp.keys())
    assert len(fp) == 3
    for i in range(2):
        entry_name = core.default_entry_template.format(base="new_entry", index=i + 1)
        entry = fp[entry_name]
        assert len(entry) == len(datasets)
        assert set(entry.keys()) == set(dset["name"] for dset in datasets)
        # this will fail if iteration is not in order of creation
        for dset, d in zip(datasets, entry.values()):
            assert d.shape == dset["data"].shape
            assert not arf.is_entry(d)


def test_copy_entry(src_arf_file, tmp_path):
    tgt_file = tmp_path / "output.arf"

    core.copy_entries(tgt_file, [src_arf_file / "entry"])
    with arf.open_file(tgt_file, "r") as fp:
        entry = fp["/entry"]
        assert len(entry) == len(datasets)
        assert set(entry.keys()) == set(dset["name"] for dset in datasets)
        # this will fail if iteration is not in order of creation
        for dset, d in zip(datasets, entry.values()):
            assert d.shape == dset["data"].shape
            assert not arf.is_entry(d)


def test_copy_nonexistent_things(src_arf_file, tmp_path):
    tgt_file = tmp_path / "output.arf"
    core.copy_entries(tgt_file, ["no_such_file.arf"])
    core.copy_entries(tgt_file, [src_arf_file / "no_such_entry"])
    fp = arf.open_file(tgt_file, "r")
    assert len(fp) == 0


def test_list_non_existent_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        core.list_entries(tmp_path / "no_such_file.arf")


def test_list_all_entries(src_arf_file):
    # doesn't test the actual output, just make sure the function runs
    core.list_entries(src_arf_file)


def test_list_an_entry(src_arf_file):
    # doesn't test the actual output, just make sure the function runs
    core.list_entries(src_arf_file, ["entry"])


def test_toplevel_attributes(src_arf_file, tmp_path):
    test_text = "abracadabra"
    tmp_text = tmp_path / "my_text.txt"
    tmp_text.write_text(test_text)
    core.write_toplevel_attribute(src_arf_file, [tmp_text])
    with arf.open_file(src_arf_file, "r") as fp:
        assert fp.attrs[f"user_{tmp_text.name}"] == test_text
    # just test that the read function works
    core.read_toplevel_attribute(src_arf_file, ["my_text.txt"])


@pytest.mark.skipif(sys.platform == "win32", reason="Test does not run on Windows")
def test_repack(src_arf_file):
    core.repack_file(src_arf_file, compress=9)


def test_repack_nonexistent_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        core.repack_file(tmp_path / "no_such_file.arf")
