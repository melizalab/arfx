# -*- coding: utf-8 -*-
# -*- mode: python -*-
import unittest
from distutils import version

import os
import numpy as np
from arfx import pcmio


class TestPcm(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.data = np.random.randint(-(2 ** 15), 2 ** 15, 1000).astype("h")
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.pcm")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir)

    def readwrite(self, dtype, nchannels):

        with pcmio.pcmfile(
            self.test_file, mode="w+", sampling_rate=20000, dtype="h", nchannels=1
        ) as fp:
            self.assertEqual(fp.filename, self.test_file)
            self.assertEqual(fp.sampling_rate, 20000)
            self.assertEqual(fp.mode, "r+")
            self.assertEqual(fp.nchannels, 1)
            self.assertEqual(fp.dtype.char, "h")

            fp.write(self.data)
            self.assertEqual(fp.nframes, self.data.size)
            self.assertTrue(np.all(fp.read() == self.data))

        with pcmio.pcmfile(self.test_file, mode="r") as fp:
            self.assertEqual(fp.filename, self.test_file)
            self.assertEqual(fp.sampling_rate, 20000)
            self.assertEqual(fp.mode, "r")
            self.assertEqual(fp.nchannels, 1)
            self.assertEqual(fp.dtype.char, "h")
            self.assertEqual(fp.nframes, self.data.size)
            read = fp.read()
            self.assertTrue(np.all(read == self.data))

    def test01_readwrite(self):
        dtypes = ("b", "h", "i", "l", "f", "d")
        nchannels = (1, 2, 8)
        for dtype in dtypes:
            for nc in nchannels:
                with self.subTest(dtype=dtype, channels=nc):
                    self.readwrite(dtype, nc)

    def test00_badmode(self):
        with self.assertRaises(ValueError):
            pcmio.pcmfile(self.test_file, mode="z")

    def test02_append(self):
        dtype = "h"
        to_write = np.random.randint(-(2 ** 15), 2 ** 15, (1000,)).astype(dtype)
        with pcmio.pcmfile(self.test_file, mode="w", sampling_rate=20000) as ofp:
            ofp.write(to_write)
            ofp.write(to_write)

        expected = np.concatenate((to_write, to_write))
        with pcmio.pcmfile(self.test_file, mode="r") as fp:
            self.assertEqual(fp.mode, "r")
            data = fp.read(memmap="r")
            self.assertEqual(data.dtype, to_write.dtype)
            self.assertEqual(data.size, expected.size)
            self.assertTrue(np.all(data == expected))
