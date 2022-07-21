# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
Read and write raw binary format files

Copyright (C) 2012 Dan Meliza <dan // AT // meliza.org>
Created 2012-03-29
"""
from ewave import rescale


class pcmfile:
    """Provides access to sampled data in raw binary format

    Raw binary files store sampled data as a continuous little-endian array. The
    data type and channel layout must be known in order to correctly map the
    contents in and out of memory.

    If the file contains more than one channel, the layout must be T x N, where
    N is the number of channels and T is the number of samples.

    file:          the path of the file to open, or an open file-like (binary-mode) object
    mode:          the mode to open the file. if already open, uses the file's handle
    sampling_rate: specify the sampling rate of the data. this has no effect on the file.
    dtype:         specify the data type using numpy character codes.
                   'b','h','i','l':  8,16,32,64-bit PCM
                   'f','d':  32,64-bit IEEE float
    byteorder:     specify byte order ('<' corresponds to little-endian, the default)
    nchannels:     specify the number of channels

    additional keyword arguments are ignored

    """

    def __init__(
        self,
        file,
        mode="r",
        sampling_rate=20000,
        dtype="h",
        nchannels=1,
        byteorder="<",
        **kwargs
    ):
        from numpy import dtype as ndtype

        # validate arguments
        self._dtype = ndtype(dtype).newbyteorder(byteorder)
        self._nchannels = int(nchannels)
        self._framerate = int(sampling_rate)

        if hasattr(file, "read"):
            self.fp = file
        else:
            if mode not in ("r", "r+", "w", "w+"):
                raise ValueError("Invalid mode (use 'r', 'r+', 'w', 'w+')")
            self.fp = open(file, mode=mode + "b")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fp.close()

    @property
    def filename(self):
        """The path of the file"""
        return self.fp.name

    @property
    def mode(self):
        """The mode for the file"""
        return self.fp.mode.replace("b", "")

    @property
    def sampling_rate(self):
        return self._framerate

    @property
    def nchannels(self):
        return self._nchannels

    @property
    def nframes(self):
        # not sure how this will behave with memmap
        pos = self.fp.tell()
        self.fp.seek(0, 2)
        nbytes = self.fp.tell()
        self.fp.seek(pos, 0)
        return nbytes // (self.dtype.itemsize * self.nchannels)

    @property
    def dtype(self):
        """Data storage type"""
        return self._dtype

    def __repr__(self):
        return (
            "<open %s.%s %s, mode='%s', dtype='%s', channels=%d, sampling rate %d at %s>"
            % (
                self.__class__.__module__,
                self.__class__.__name__,
                self.filename,
                self.mode,
                self.dtype,
                self.nchannels,
                self.sampling_rate,
                hex(id(self)),
            )
        )

    def flush(self):
        """flush data to disk"""
        if hasattr(self, "fp") and not self.fp.closed:
            self.fp.flush()
        return self

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

        if self.mode == "w":
            raise IOError("file is write-only")
        if self.mode in ("r+", "w+"):
            self.fp.flush()
        # find offset
        if frames is None:
            frames = self.nframes - offset
        if memmap:
            A = mmap(
                self.fp,
                offset=offset,
                dtype=self._dtype,
                mode=memmap,
                shape=(frames, self.nchannels),
            )
        else:
            pos = self.fp.tell()
            self.fp.seek(offset)
            A = fromfile(self.fp, dtype=self._dtype, count=frames * self.nchannels)
            self.fp.seek(pos)

        if self.nchannels > 1:
            nsamples = (A.size // self.nchannels) * self.nchannels
            A = A[:nsamples]
            A.shape = (nsamples // self.nchannels, self.nchannels)
        return A.squeeze()

    def write(self, data, scale=True):
        """Write data to the end of the file

        This function can be called repeatedly to store data in chunks. Each chunk
        is appended at the end of the file.

        - data : input data, in any form that can be converted to an array with
                 the file's dtype. Data are silently coerced into an array whose
                 shape matches the number of channels in the file. This means
                 the caller is responsible for checking dimensions in
                 multichannel files.

        - scale : if True, data are rescaled so that their maximum range matches
                    that of the file's encoding. If not, the raw values are
                    used, which can result in clipping.

        """
        from numpy import asarray

        if self.mode == "r":
            raise IOError("file is read-only")

        if scale:
            data = rescale(data, self._dtype)
        else:
            data = asarray(data, self._dtype)

        self.fp.seek(0, 2)
        self.fp.write(memoryview(data).cast("B"))
        return self


# Variables:
# End:
