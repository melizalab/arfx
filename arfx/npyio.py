# -*- coding: utf-8 -*-
# -*- mode: python -*-
""" Read and write numpy files

Implementation is based on https://numpy.org/devdocs/reference/generated/numpy.lib.format.html
"""
from .io import extended_shape


class npyfile:
    """Provides access to sampled data in numpy format

    For reading data, this is a very thin wrapper around numpy.load. For writing
    data, it is possible to append data to a open file as long as the dtype and
    dimensions are compatible (i.e the same except for the first dimension).

    If the file contains more than one channel, the layout must be T x N, where
    N is the number of channels and T is the number of samples.

    file:          the path of the file to open
    mode:          the mode to open the file ('r' or 'w')
    sampling_rate: specify the sampling rate of the data. this has no effect on the file.

    additional keyword arguments are ignored

    """

    def __init__(self, file, mode="r", sampling_rate=20000, **kwargs):
        from numpy import load

        self.sampling_rate = sampling_rate
        self.filename = file
        self.mode = mode

        if mode == "w":
            self.fp = open(file, mode=mode + "b")
        elif mode == "r":
            self.data = load(file, mmap_mode="r")
        else:
            raise ValueError("Invalid mode (use 'r' or 'w')")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.mode == "w":
            self.fp.close()

    def read(self, mmap_mode=None):
        if self.mode != "r":
            raise IOError("attempted to read from a file opened in write mode")
        return self.data

    def write(self, data):
        from numpy import isfortran, save
        from numpy.lib.format import header_data_from_array_1_0

        if isfortran(data):
            raise ValueError("data must be C-contiguous (row-major order)")
        if self.mode != "w":
            raise IOError("attempted to write to file opened in read-only mode")
        if self.fp.seek(0, 2) == 0:
            # empty file
            self._write_descr = header_data_from_array_1_0(data)
            self._write_header()
            self.fp.write(memoryview(data).cast("B"))
        else:
            if self._write_descr["descr"] != header_data_from_array_1_0(data)["descr"]:
                raise ValueError(
                    "data type is not compatible with previously written data"
                )
            self._write_descr["shape"] = tuple(
                extended_shape(self._write_descr["shape"], data.shape)
            )
            self.fp.write(memoryview(data).cast("B"))
            self.fp.seek(0, 0)
            self._write_header()

    def _write_header(self):
        """Write a well-padded header to the file"""
        # this is a reimplemntation of npf._wrap_header that pads the header to maximum
        import numpy.lib.format as npf
        import struct

        ARRAY_ALIGN = 2**16
        version = (1, 0)
        fmt, encoding = npf._header_size_info[version]
        header = repr(self._write_descr).encode(encoding)
        hlen = len(header) + 1
        padlen = ARRAY_ALIGN - (
            (npf.MAGIC_LEN + struct.calcsize(fmt) + hlen) % ARRAY_ALIGN
        )
        try:
            header_prefix = npf.magic(*version) + struct.pack(fmt, hlen + padlen)
        except struct.error:
            msg = "Header length {} too big for version={}".format(hlen, version)
            raise ValueError(msg)
        self.fp.seek(0)
        self.fp.write(header_prefix)
        self.fp.write(header)
        self.fp.write(b" " * padlen)
        self.fp.write(b"\n")
