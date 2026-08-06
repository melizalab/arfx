# -*- mode: python -*-
"""Fixtures shared across the arfx test suite.

DATASETS covers the dataset shapes arfx has to handle -- 1-D and 2-D sampled
data, a spike train on a sample timebase, an empty dataset, and a marked point
process -- so that a test which walks every dataset in an entry exercises all
of the branches in core.dataset_properties and the arf.is_* predicates.

Modules that need the raw specs import DATASETS directly; everything else goes
through the fixtures.
"""

import time

import arf
import numpy as np
import pytest
from numpy.random import randint, randn

from arfx import io

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
