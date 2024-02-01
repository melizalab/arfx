# -*- coding: utf-8 -*-
# -*- mode: python -*-

import numpy as np
import pytest

from arfx import mdaio


@pytest.fixture()
def test_file(tmp_path):
    return tmp_path / "test.mda"


def test_badmode(test_file):
    with pytest.raises(ValueError):
        mdaio.mdafile(test_file, mode="z")


def test_readwrite(test_file):
    dtype = "f"
    to_write = np.random.randint(-(2**15), 2**15, (1000, 2)).astype(dtype)
    with mdaio.mdafile(test_file, mode="w", sampling_rate=20000) as ofp:
        assert ofp.filename == test_file
        assert ofp.sampling_rate == 20000
        assert ofp.mode == "w"
        ofp.write(to_write)

    with mdaio.mdafile(test_file, mode="r") as fp:
        assert fp.filename == test_file
        assert fp.sampling_rate == 20000
        assert fp.mode == "r"
        data = fp.read(memmap="r")
        assert data.dtype == to_write.dtype
        assert data.size == to_write.size
        assert np.all(data == to_write)


def test_append(test_file):
    dtype = "h"
    to_write = np.random.randint(-(2**15), 2**15, (1000, 2)).astype(dtype)
    with mdaio.mdafile(test_file, mode="w", sampling_rate=20000) as ofp:
        ofp.write(to_write)
        ofp.write(to_write)

    expected = np.concatenate((to_write, to_write))
    with mdaio.mdafile(test_file, mode="r") as fp:
        assert fp.mode == "r"
        data = fp.read(memmap="r")
        assert data.dtype == to_write.dtype
        assert data.size == expected.size
        assert np.all(data == expected)


def test_appendwrongshape(test_file):
    dtype = "h"
    to_write_1 = np.zeros((1000, 2), dtype=dtype)
    to_write_2 = np.zeros((1000,), dtype=dtype)
    with mdaio.mdafile(test_file, mode="w", sampling_rate=20000) as ofp:
        ofp.write(to_write_1)
        with pytest.raises(ValueError):
            ofp.write(to_write_2)


def test_appendwrongdtype(test_file):
    to_write_1 = np.zeros((1000,), dtype="h")
    to_write_2 = np.zeros((1000,), dtype="f")
    with mdaio.mdafile(test_file, mode="w", sampling_rate=20000) as ofp:
        ofp.write(to_write_1)
        with pytest.raises(ValueError):
            ofp.write(to_write_2)
