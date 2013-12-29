# -*- coding: utf-8 -*-
# -*- mode: python -*-

from __future__ import division
from __future__ import unicode_literals

from nose.tools import *

import os
import numpy as nx
from arfx import pcmio

test_file = None

def setup():
    import tempfile
    global temp_dir, test_file, data
    data = nx.random.randint(-2**15, 2**15, 1000).astype('h')
    temp_dir = tempfile.mkdtemp()
    test_file = os.path.join(temp_dir, "test.pcm")


def teardown():
    import shutil
    shutil.rmtree(temp_dir)


def test_readwrite():

    with pcmio.pcmfile(test_file, mode="w+", sampling_rate=20000, dtype='h', nchannels=1) as fp:
        assert_equal(fp.filename, test_file)
        assert_equal(fp.sampling_rate, 20000)
        assert_equal(fp.mode, "w")
        assert_equal(fp.nchannels, 1)
        assert_equal(fp.dtype.char, 'h')

        fp.write(data)
        assert_equal(fp.nframes, data.size)
        assert_true(nx.all(fp.read() == data))

    with pcmio.pcmfile(test_file, mode="r") as fp:
        assert_equal(fp.filename, test_file)
        assert_equal(fp.sampling_rate, 20000)
        assert_equal(fp.mode, "r")
        assert_equal(fp.nchannels, 1)
        assert_equal(fp.dtype.char, 'h')
        assert_equal(fp.nframes, data.size)

        read = fp.read()
        assert_true(nx.all(read == data))

        f = raises(IOError)(fp.read)
        f("some garbage")


@raises(ValueError)
def test_badmode():
    pcmio.pcmfile(test_file, mode="z")


@raises(ValueError)
def test_badchans():
    pcmio.pcmfile(test_file, mode="w", nchannels=2)


