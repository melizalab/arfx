# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
Read and write mountainlab binary format files

Copyright (C) 2020 Dan Meliza <dan // AT // meliza.org>

"""
import numpy as np

from .io import extended_shape

NUM_DTYPE = {
    -2: "uint8",
    -3: "float32",
    -4: "int16",
    -5: "int32",
    -6: "uint16",
    -7: "float64",
    -8: "uint32",
}

DTYPE_NUM = {v: k for k, v in NUM_DTYPE.items()}


class mdafile:
    """Provides access to sampled data in mountainlab binary format

    The mountainlab format is a raw binary file with a simple header. If the
    file contains more than one channel, the layout must be T x N, where N is
    the number of channels and T is the number of samples. Note that the
    official documentation has this backwards, perhaps because they're coming
    from MATLAB (i.e. Fortran) world.

    The returned object may be used as a context manager, and will close the
    underlying file when the context exits. Data can be appended to files opened
    for writing as long as the number of channels and data type remain the same.

    file:          the path of the file to open
    mode:          the mode to open the file ('r' or 'w')
    sampling_rate: specify the sampling rate of the data. this has no effect on the file.

    additional keyword arguments are ignored

    """

    def __init__(self, file, mode="r", sampling_rate=20000, dtype="h", **kwargs):
        # validate arguments
        self.sampling_rate = sampling_rate
        self.filename = file
        self.mode = mode

        if mode in ("w", "r"):
            self.fp = open(file, mode=mode + "b")
        else:
            raise ValueError("Invalid mode (use 'r' or 'w')")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fp.close()

    def _read_header(self):
        import struct

        self.fp.seek(0, 0)
        dt_code, itemsize, ndims = struct.unpack(b"<lll", self.fp.read(12))
        try:
            self.dtype = np.dtype(NUM_DTYPE[dt_code])
        except KeyError:
            raise ValueError("invalid data type code in header")
        assert (
            self.dtype.itemsize == itemsize
        ), "item size does not match header data type"
        if ndims < 0:
            ndims = abs(ndims)
            fmt = b"<" + b"Q" * ndims
            shape = struct.unpack(fmt, self.fp.read(8 * ndims))
        else:
            fmt = b"<" + b"L" * ndims
            shape = struct.unpack(fmt, self.fp.read(4 * ndims))
        if ndims > 2:
            raise ValueError("data must be no more than 2 dimensions")
        self.nchannels, self.nframes = shape

    def _write_header(self, dtype, shape):
        import struct

        # TODO: use 64-bit dimensions, once mountainlab actually supports this
        self.dtype = dtype
        if len(shape) > 2:
            raise ValueError("data must be no more than 2 dimensions")
        self.nframes = shape[0]
        if len(shape) == 1:
            shape = (shape[0], 1)
        self.nframes, self.nchannels = shape
        header = struct.pack(
            b"<lllLL", DTYPE_NUM[dtype.name], dtype.itemsize, 2, *reversed(shape)
        )
        self.fp.seek(0, 0)
        self.fp.write(header)

    def read(self, frames=None, offset=0, memmap="c"):
        """
        Return contents of file. Default is is to memmap the data in
        copy-on-write mode, which means read operations are delayed
        until the data are actually accessed or modified.

        - frames: number of frames to return. None for all the frames in the file
        - offset: start read at specific frame
        - memmap: if False, reads the whole file into memory at once; if not, returns
                  a numpy.memmap object using this value as the mode argument. 'c'
                  corresponds to copy-on-write; use 'r+' to write changes to disk. Be
                  warned that 'w' modes may corrupt data.
        """
        from numpy import memmap as mmap
        from numpy import fromfile

        if self.mode != "r":
            raise IOError("attempted to read from a file opened in write mode")
        header = self._read_header()
        if frames is None:
            frames = self.nframes - offset
        offset = self.fp.tell() + offset * self.dtype.itemsize
        if self.nchannels > 1:
            shape = (frames, self.nchannels)
        else:
            shape = (frames,)
        if memmap:
            A = mmap(self.fp, offset=offset, dtype=self.dtype, mode=memmap, shape=shape)
        else:
            self.fp.seek(offset, 0)
            A = fromfile(self.fp, dtype=self.dtype, count=frames * self.nchannels)
            if self.nchannels > 1:
                A.shape = shape
        return A

    def write(self, data):
        """Write data to the end of the file

        This function can be called repeatedly to store data in chunks. Each chunk
        is appended at the end of the file.

        - data : input data, in any form that can be converted to an array with
                 the file's dtype. Data are silently coerced into an array whose
                 shape matches the number of channels in the file. This means
                 the caller is responsible for checking dimensions in
                 multichannel files.
        """
        from numpy import isfortran

        if isfortran(data):
            raise ValueError("data must be C-contiguous (row-major order)")
        if self.mode != "w":
            raise IOError("attempted to write to file opened in read-only mode")
        if self.fp.seek(0, 2) == 0:
            # empty file
            self._write_header(data.dtype, data.shape)
        else:
            if self.dtype != data.dtype:
                raise ValueError(
                    "data type is not compatible with previously written data"
                )
            shape = tuple(extended_shape((self.nframes, self.nchannels), data.shape))
            self._write_header(data.dtype, shape)
            self.fp.seek(0, 2)

        self.fp.write(memoryview(data).cast("B"))
        return self


# Variables:
# End:
