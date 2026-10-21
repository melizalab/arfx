# -*- mode: python -*-
"""Metadata reader for neurobank: ARF files made by arfx-oephys

Registered as an entry point in the group ``nbank.metadata_readers``, for the
``extracellular-arf`` and ``eeg-arf`` dtypes. The reader checks that a file is
what `arfx-oephys` writes and returns the metadata for the registry: the
attributes given with -k (and --experimenter, --animal), and the versions of
arfx-oephys that wrote the entries. It does not import neurobank.

"""

import math
from pathlib import Path
from typing import Any

import arf
import h5py
import numpy as np

#: entry attribute listing the names of the attributes that came from -k
USER_ATTRIBUTES = "user_attributes"

# Files written before arfx-oephys recorded USER_ATTRIBUTES can't say which of
# their attributes were given by the user, so only these are taken.
LEGACY_KEYS = ("experimenter", "animal", "bird", "experiment", "protocol")

# Names used before the registry's were agreed, and ARF's own attribute for the
# program that wrote the entry (e.g. "org.meliza.arfx/arfx-oephys 3.0.2")
RENAMED = {
    "bird": "animal",
    "experiment": "protocol",
    "entry_creator": "processed_by",
}

# An entry with a different value for one of these (or without it, when others
# have it) is a file that mixes recordings; any other attribute may vary between
# entries and is then left out.
IDENTITY_KEYS = ("experimenter", "animal")


class _Unstorable(Exception):
    """A value that JSON and the registry can't hold (NaN, infinity)"""


def _json_value(value: Any) -> Any:
    """Convert a value read from an HDF5 attribute to a JSON-serializable one"""
    if isinstance(value, bytes):  # includes np.bytes_
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise _Unstorable
    if isinstance(value, (str, bool, int, float)):
        return value
    raise _Unstorable


def _entry_metadata(entry: h5py.Group) -> dict[str, Any]:
    entry_name = entry.name.lstrip("/")
    names = entry.attrs.get(USER_ATTRIBUTES)
    if names is None:
        names = LEGACY_KEYS
    else:
        names = [_json_value(n) for n in np.atleast_1d(names)]
    # entry_creator is ARF's, not the user's, so is never in the list
    names = [*names, "entry_creator"]
    metadata = {}
    for name in names:
        if name not in entry.attrs:
            continue
        try:
            value = _json_value(entry.attrs[name])
        except _Unstorable:
            continue
        metadata[name] = value
    for old, new in RENAMED.items():
        if old in metadata:
            value = metadata.pop(old)
            if new in metadata and metadata[new] != value and old != "entry_creator":
                raise ValueError(
                    f"entry '{entry_name}' has {old}={value!r} and "
                    f"{new}={metadata[new]!r}"
                )
            metadata.setdefault(new, value)
    if not isinstance(metadata.get("processed_by", ""), str):
        raise ValueError(
            f"entry '{entry_name}' has a processed_by that isn't a string: "
            f"{metadata['processed_by']!r}"
        )
    return metadata


def _check_continuous(entry: h5py.Group) -> None:
    for dset in entry.values():
        if (
            isinstance(dset, h5py.Dataset)
            and dset.dtype.names is None
            and "sampling_rate" in dset.attrs
        ):
            rate = dset.attrs["sampling_rate"]
            if (
                np.issubdtype(np.asarray(rate).dtype, np.number)
                and np.size(rate) == 1
                and np.isfinite(rate)
                and rate > 0
            ):
                return
    raise ValueError(
        f"entry '{entry.name.lstrip('/')}' has no sampled dataset with a "
        "sampling rate; it was not made by arfx-oephys"
    )


def read_metadata(path: Path) -> dict[str, Any]:
    """Return the registry metadata of an ARF file written by arfx-oephys

    Raises ValueError if the file is not such a file, or if its entries
    disagree about the experimenter or the animal.
    """
    path = Path(path)
    try:
        fp = h5py.File(path, "r")
    except OSError as e:
        # also raised for a file that is locked or can't be read
        raise ValueError(f"could not open as an HDF5 file ({e})") from None
    with fp:
        if "arf_version" not in fp.attrs:
            raise ValueError("not an ARF file: no arf_version attribute")
        entries = [name for name in fp if arf.is_entry(fp[name])]
        if not entries:
            raise ValueError("ARF file has no entries")
        per_entry = []
        for name in entries:
            _check_continuous(fp[name])
            per_entry.append((name, _entry_metadata(fp[name])))

    first_name, first = per_entry[0]
    keys = set().union(*(m.keys() for _, m in per_entry))
    result = {}
    for key in sorted(keys):
        if key == "processed_by":
            # every version that wrote an entry processed the file
            seen = dict.fromkeys(m[key] for _, m in per_entry if key in m)
            result[key] = "; ".join(seen)
        elif all(key in m and m[key] == first.get(key) for _, m in per_entry):
            result[key] = first[key]
        elif key in IDENTITY_KEYS:
            for name, m in per_entry:
                if m.get(key) != first.get(key):
                    raise ValueError(
                        f"entries disagree about {key}: "
                        f"'{first_name}' has {first.get(key)!r}, "
                        f"'{name}' has {m.get(key)!r}"
                    )
    return result
