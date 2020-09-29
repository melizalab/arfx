# -*- coding: utf-8 -*-
# -*- mode: python -*-
import unittest
from distutils import version

import os
import numpy as nx
from arfx import pcmio

class TestPcm(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.data = nx.random.randint(-2**15, 2**15, 1000).astype('h')
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.pcm")


    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)


    def readwrite(self, dtype, nchannels):

        with pcmio.pcmfile(self.test_file, mode="w+", sampling_rate=20000, dtype='h', nchannels=1) as fp:
            assert_equal(fp.filename, self.test_file)
            assert_equal(fp.sampling_rate, 20000)
            assert_equal(fp.mode, "w")
            assert_equal(fp.nchannels, 1)
            assert_equal(fp.dtype.char, 'h')

            fp.write(self.data)
            assert_equal(fp.nframes, self.data.size)
            assert_true(nx.all(fp.read() == self.data))

        with pcmio.pcmfile(self.test_file, mode="r") as fp:
            assert_equal(fp.filename, self.test_file)
            assert_equal(fp.sampling_rate, 20000)
            assert_equal(fp.mode, "r")
            assert_equal(fp.nchannels, 1)
            assert_equal(fp.dtype.char, 'h')
            assert_equal(fp.nframes, self.data.size)

            read = fp.read()
            assert_true(nx.all(read == self.data))

            with self.assertRaises(IOError):
                fp.read("some garbage")


    def test_readwrite(self):
        dtypes = ('b', 'h', 'i', 'l', 'f', 'd')
        nchannels = (1, 2, 8)
        for dtype in dtypes:
            for nc in nchannels:
                yield self.readwrite, dtype, nc


    def test_badmode(self):
        with self.assertRaises(ValueError):
            pcmio.pcmfile(self.test_file, mode="z")
