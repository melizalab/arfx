# -*- coding: utf-8 -*-
# -*- mode: python -*-
import unittest
from distutils import version

import os
import numpy as np
from arfx import npyio


class TestNpy(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.npy")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir)

    def test00_badmode(self):
        with self.assertRaises(ValueError):
            npyio.npyfile(self.test_file, mode="z")

    def test01_readwrite(self):
        dtype = "f"
        to_write = np.random.randint(-(2 ** 15), 2 ** 15, (1000, 2)).astype(dtype)
        with npyio.npyfile(self.test_file, mode="w", sampling_rate=20000) as ofp:
            self.assertEqual(ofp.filename, self.test_file)
            self.assertEqual(ofp.sampling_rate, 20000)
            self.assertEqual(ofp.mode, "w")
            ofp.write(to_write)

        with npyio.npyfile(self.test_file, mode="r") as fp:
            self.assertEqual(fp.filename, self.test_file)
            self.assertEqual(fp.sampling_rate, 20000)
            self.assertEqual(fp.mode, "r")
            data = fp.read(mmap_mode="r")
            self.assertEqual(data.dtype, to_write.dtype)
            self.assertEqual(data.size, to_write.size)
            self.assertTrue(np.all(data == to_write))

    def test02_append(self):
        dtype = "h"
        to_write = np.random.randint(-(2 ** 15), 2 ** 15, (1000, 2)).astype(dtype)
        with npyio.npyfile(self.test_file, mode="w", sampling_rate=20000) as ofp:
            ofp.write(to_write)
            ofp.write(to_write)

        expected = np.concatenate((to_write, to_write))
        with npyio.npyfile(self.test_file, mode="r") as fp:
            self.assertEqual(fp.mode, "r")
            data = fp.read(mmap_mode="r")
            self.assertEqual(data.dtype, to_write.dtype)
            self.assertEqual(data.size, expected.size)
            self.assertTrue(np.all(data == expected))

    def test03_appendwrongshape(self):
        dtype = "h"
        to_write_1 = np.zeros((1000, 2), dtype=dtype)
        to_write_2 = np.zeros((1000,), dtype=dtype)
        with npyio.npyfile(self.test_file, mode="w", sampling_rate=20000) as ofp:
            ofp.write(to_write_1)
            with self.assertRaises(ValueError):
                ofp.write(to_write_2)

    def test04_appendwrongdtype(self):
        to_write_1 = np.zeros((1000,), dtype="h")
        to_write_2 = np.zeros((1000,), dtype="f")
        with npyio.npyfile(self.test_file, mode="w", sampling_rate=20000) as ofp:
            ofp.write(to_write_1)
            with self.assertRaises(ValueError):
                ofp.write(to_write_2)
