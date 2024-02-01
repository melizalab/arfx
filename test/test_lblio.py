# -*- coding: utf-8 -*-
# -*- mode: python -*-
from io import StringIO

import pytest

from arfx import lblio

lbl_good = """\
signal feasd
type 0
color 121
font *-fixed-bold-*-*-*-15-*-*-*-*-*-*-*
separator ;
nfields 1
#
   15.445851  121 A
   15.520200  121 a-0
   15.595700  121 a-1
   15.747526  121 a-0
   15.818300  121 a-1
   15.928394  121 a-0
   15.991940  121 a-1
   16.053200  121 b-0
   16.192361  121 b-1
   16.230769  121 c-0
   16.350176  121 c-1
   16.395300  121 d-0
   16.740300  121 d-1
   16.847382  121 V
   17.010093  121 C
   19.012345  121 αγ
   20.123183  121 long-name-0
   20.353885  121 long-name-1
"""  # noqa: RUF001
lbl_events = ("A", "V", "C", "αγ")  # noqa: RUF001
lbl_intervals = ("a", "b", "c", "d", "long-name")


@pytest.fixture(scope="module")
def labels():
    return lblio.read(StringIO(lbl_good))


def test_event_count(labels):
    assert labels.shape == (11,)


def test_event_array_names(labels):
    assert labels.dtype.names == ("name", "start", "stop")


def test_event_names(labels):
    assert set(labels["name"]) == set(lbl_events + lbl_intervals)


def test_events_and_intervals(labels):
    for event in labels:
        if event["name"] in lbl_events:
            assert event["start"] == event["stop"]
        else:
            assert event["start"] != event["stop"]


def test_bad_header():
    lbl_bad_header = """\
        signal feasd
        type 0
        color 121
        font *-fixed-bold-*-*-*-15-*-*-*-*-*-*-*
        nfields 1
        #
           15.445851  121 A
        """
    with pytest.raises(ValueError):
        _ = lblio.read(StringIO(lbl_bad_header))


def test_missing_interval_opener():
    lbl = """\
        signal feasd
        type 0
        color 121
        font *-fixed-bold-*-*-*-15-*-*-*-*-*-*-*
        separator ;
        nfields 1
        #
           15.445851  121 A
           15.595700  121 a-1
           15.747526  121 a-0
           17.010093  121 C
        """
    with pytest.raises(ValueError):
        _ = lblio.read(StringIO(lbl))


def test_missing_interval_close():
    lbl = """\
        signal feasd
        type 0
        color 121
        font *-fixed-bold-*-*-*-15-*-*-*-*-*-*-*
        separator ;
        nfields 1
        #
           15.445851  121 A
           15.595700  121 a-0
           15.747526  121 a-0
           17.010093  121 C
        """
    with pytest.raises(ValueError):
        _ = lblio.read(StringIO(lbl))
