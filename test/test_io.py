# -*- coding: utf-8 -*-
# -*- mode: python -*-
import pytest

from arfx import io


def test_included_plugins():
    formats = io.list_plugins()
    assert set(formats) == {".dat", ".mda", ".npy", ".pcm", ".wav"}


def test_open_mda(tmp_path):
    tmp_file = tmp_path / "test.mda"
    fp = io.open(tmp_file, "w", sampling_rate=20000)


def test_open_pcm(tmp_path):
    tmp_file = tmp_path / "test.pcm"
    fp = io.open(tmp_file, "w", sampling_rate=20000)


def test_open_npy(tmp_path):
    tmp_file = tmp_path / "test.npy"
    fp = io.open(tmp_file, "w", sampling_rate=20000)


def test_open_wav(tmp_path):
    tmp_file = tmp_path / "test.wav"
    fp = io.open(tmp_file, "w", sampling_rate=20000)


def test_unsupported_format(tmp_path):
    tmp_file = tmp_path / "test.blah"
    with pytest.raises(ValueError):
        fp = io.open(tmp_file, "w")
