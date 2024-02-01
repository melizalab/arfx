# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""Read and write lbl format files. 

lbl is a format developed by the Margoliash lab. It's not a very good or
flexible format, but it's what the venerable data inspection and labeling
program aplot uses, so it's important to be able to read it.

This code is adapted from https://github.com/kylerbrown/lbl

The lbl standard uses the following format:

    7 lines of garbage (the header)

After the header, each line represents an entry with space separated elements.

    The first element is a floating point time stamp
    the second element is garbage (the number 121)
    the third element is the label or name.

Copyright (C) 2022 Dan Meliza <dan // AT // meliza.org>

"""


class lblfile:
    """Reader for a Margoliash lab `lbl` file."""

    def __init__(self, fname, mode="r", **kwargs):
        self.fp = open(fname, mode + "t")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fp.close()

    def read(self):
        """Read the contents of the file, returning a structured numpy array"""
        return _read(self.fp)

    def write(fp, labels):
        raise NotImplementedError


def read(fp):
    """Parses the contents of a fp (a file object or stream containing an lbl
    file) and returns contents as a structured numpy array

    """
    from numpy import array

    # read and discard header
    for _i in range(7):
        line = fp.readline()
        if len(line) == 0:
            raise ValueError("Invalid lbl file - header has empty lines")
    labels = []
    unmatched = {}
    for i, line in enumerate(fp):
        time, _, label = line.split()
        label = label.strip()
        try:
            time = float(time)
        except ValueError as e:
            raise ValueError(f"Parse error in line {i + 8}: {e}") from e
        if label.endswith("-0"):
            base = label[:-2]
            if base in unmatched:
                raise ValueError(
                    f"Parse error in line {i + 8}: opening interval "
                    f"for {base} but there is an unclosed opening at {unmatched[base]} s"
                )
            unmatched[base] = time
        elif label.endswith("-1"):
            base = label[:-2]
            try:
                start_time = unmatched.pop(base)
                labels.append((base, start_time, time))
            except KeyError as err:
                raise ValueError(
                    f"Parse error in line {i + 8}: closing interval "
                    f"for {base} but no opening"
                ) from err
        else:
            labels.append((label, time, time))
    maxlen = max(len(lbl[0]) for lbl in labels)
    dtype = [("name", f"U{maxlen}"), ("start", "f8"), ("stop", "f8")]
    return array(labels, dtype=dtype)


_read = read
