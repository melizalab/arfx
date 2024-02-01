# -*- coding: utf-8 -*-
# -*- mode: python -*-
import numpy as np
import pytest

from arfx import pcmio


@pytest.fixture
def test_data():
    return np.random.randint(-(2**15), 2**15, 1000).astype("h")


@pytest.fixture
def test_file(tmp_path):
    return tmp_path / "test.pcm"


def readwrite(test_file, test_data, dtype, nchannels):
    with pcmio.pcmfile(
        test_file, mode="w+", sampling_rate=20000, dtype="h", nchannels=1
    ) as fp:
        assert fp.filename == str(test_file)
        assert fp.sampling_rate == 20000
        assert fp.mode == "r+"
        assert fp.nchannels == 1
        assert fp.dtype.char == "h"

        fp.write(test_data)
        assert fp.nframes == test_data.size
        assert np.all(fp.read() == test_data)

    with pcmio.pcmfile(test_file, mode="r") as fp:
        assert fp.filename == str(test_file)
        assert fp.sampling_rate == 20000
        assert fp.mode == "r"
        assert fp.nchannels == 1
        assert fp.dtype.char == "h"
        assert fp.nframes == test_data.size
        read = fp.read()
        assert np.all(read == test_data)


@pytest.mark.parametrize("dtype", ["b", "h", "i", "l", "f", "d"])
@pytest.mark.parametrize("nchannels", [1, 2, 8])
def test_readwrite(dtype, nchannels, test_file, test_data):
    readwrite(test_file, test_data, dtype, nchannels)


def test_badmode(test_file):
    with pytest.raises(ValueError):
        pcmio.pcmfile(test_file, mode="z")


def test_append(test_file):
    dtype = "h"
    to_write = np.random.randint(-(2**15), 2**15, (1000,)).astype(dtype)
    with pcmio.pcmfile(test_file, mode="w", sampling_rate=20000) as ofp:
        ofp.write(to_write)
        ofp.write(to_write)

    expected = np.concatenate((to_write, to_write))
    with pcmio.pcmfile(test_file, mode="r") as fp:
        assert fp.mode == "r"
        data = fp.read(memmap="r")
        assert data.dtype == to_write.dtype
        assert data.size == expected.size
        assert np.all(data == expected)
