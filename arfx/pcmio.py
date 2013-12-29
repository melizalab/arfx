# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
Read and write PCM format files.

Copyright (C) 2012 Dan Meliza <dan // AT // meliza.org>
Created 2012-03-29
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from ewave import rescale


class pcmfile(object):

    def __init__(self, file, mode='r', sampling_rate=20000, dtype='h', nchannels=1, **kwargs):
        """Opens file for reading and/or writing. Any of the standard modes
        supported by file can be used.

        file:          the path of the file to open, or an open file-like object
        mode:          the mode to open the file. if already open, uses the file's handle
        sampling_rate: set the sampling rate of the data. this has no effect on the file.
        dtype:         set the storage format (i.e. how the data will be read)
                       'b','h','i','l':  8,16,32,64-bit PCM
                       'f','d':  32,64-bit IEEE float
        nchannels:     must be 1

        additional keyword arguments are ignored
        """
        import sys
        if sys.version > '3':
            from builtins import open
        else:
            from __builtin__ import open
        from numpy import dtype as ndtype
        # validate arguments
        self._dtype = ndtype(dtype)
        self._nchannels = int(nchannels)
        self._framerate = int(sampling_rate)
        if not nchannels == 1:
            raise ValueError(
                "More than one channel not supported by this format")

        if hasattr(file, 'read'):
            self.fp = file
        else:
            try:
                file = file.encode(sys.getfilesystemencoding())
            except (UnicodeError, LookupError):
                pass
            if mode not in ('r', 'r+', 'w', 'w+'):
                raise ValueError("Invalid mode (use 'r', 'r+', 'w', 'w+')")
            self.fp = open(file, mode=mode + 'b')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fp.close()
        return exc_val

    @property
    def filename(self):
        """ The path of the file """
        return self.fp.name

    @property
    def mode(self):
        """ The mode for the file """
        return self.fp.mode.replace('b', '')

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
        """ Data storage type """
        return self._dtype

    def __repr__(self):
        return "<open %s.%s '%s', mode '%s', dtype '%s', sampling rate %d at %s>" % \
            (self.__class__.__module__,
             self.__class__.__name__,
             self.filename,
             self.mode,
             self.dtype,
             self.sampling_rate,
             hex(id(self)))

    def flush(self):
        """ flush data to disk """
        if hasattr(self, 'fp') and not self.fp.closed:
            self.fp.flush()
        return self

    def read(self, frames=None, offset=0, memmap='c'):
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
        if self.mode == 'w':
            raise IOError('file is write-only')
        if self.mode in ('r+', 'w+'):
            self.fp.flush()
        # find offset
        if frames is None:
            frames = self.nframes - offset
        if memmap:
            A = mmap(self.fp, offset=offset, dtype=self._dtype, mode=memmap,
                     shape=frames * self.nchannels)
        else:
            pos = self.fp.tell()
            self.fp.seek(offset)
            A = fromfile(
                self.fp, dtype=self._dtype, count=frames * self.nchannels)
            self.fp.seek(pos)

        if self.nchannels > 1:
            nsamples = (A.size // self.nchannels) * self.nchannels
            A = A[:nsamples]
            A.shape = (nsamples // self.nchannels, self.nchannels)
        return A

    def write(self, data, scale=True):
        """ Write data to the file

        - data : input data, in any form that can be converted to an array with
                 the file's dtype. Data are silently coerced into an array whose
                 shape matches the number of channels in the file.

        - scale : if True, data are rescaled so that their maximum range matches
                    that of the file's encoding. If not, the raw values are
                    used, which can result in clipping.
        """
        from numpy import asarray
        if self.mode == 'r':
            raise IOError('file is read-only')

        if not scale:
            data = asarray(data, self._dtype)
        data = rescale(data, self._dtype).tostring()

        self.fp.write(data)
        return self


open = pcmfile

# Variables:
# End:
