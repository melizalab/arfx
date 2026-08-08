# -*- mode: python -*-
import pytest

from arfx import io


def test_included_plugins():
    formats = io.list_plugins()
    assert set(formats) == {".dat", ".mda", ".npy", ".pcm", ".wav"}


def test_open_mda(tmp_path):
    tmp_file = tmp_path / "test.mda"
    _fp = io.open(tmp_file, "w", sampling_rate=20000)


def test_open_pcm(tmp_path):
    tmp_file = tmp_path / "test.pcm"
    _fp = io.open(tmp_file, "w", sampling_rate=20000)


def test_open_npy(tmp_path):
    tmp_file = tmp_path / "test.npy"
    _fp = io.open(tmp_file, "w", sampling_rate=20000)


def test_open_wav(tmp_path):
    tmp_file = tmp_path / "test.wav"
    _fp = io.open(tmp_file, "w", sampling_rate=20000)


def test_unsupported_format(tmp_path):
    tmp_file = tmp_path / "test.blah"
    with pytest.raises(ValueError):
        _fp = io.open(tmp_file, "w")


def test_open_reports_a_missing_handler():
    with pytest.raises(ValueError, match="No handler defined"):
        io.open("nosuch.zzz")


@pytest.mark.parametrize("name", ["x.dat", "x.npy"])
def test_open_does_not_disguise_a_handler_error(name):
    # the try used to wrap the handler's construction as well as the lookup, so
    # a ValueError raised by the handler for its own reasons came back as
    # "No handler defined for files of type '.dat'" -- which is not only wrong
    # but points away from the actual problem
    with pytest.raises(ValueError, match="Invalid mode"):
        io.open(name, mode="q")
