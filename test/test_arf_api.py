# -*- mode: python -*-
"""Characterization tests for the arf library API that arfx depends on.

arfx is a thin CLI layer over arf, and most of the arf API surface it uses is
reached only from scripts that are otherwise lightly tested. These tests assert
on the *contract* of each arf function arfx calls -- return types, tuple arity,
ordering, exception classes, attribute encodings -- rather than on arfx-level
output.

The point is diagnostic. When arf is upgraded across a major version, run this
module first and in isolation: whatever fails here is the actual API diff.
Failures elsewhere in the suite that are not explained by a failure here are
arfx integration bugs.

Values pinned against arf 3.0.0, spec 2.2. The library version and the spec
version are on independent scales: 3.0.0 was a library major bump that shipped
against spec 2.2. Do not infer one from the other. Where a bound is derivable,
take it from supported_spec_versions() rather than writing the number twice.

Tests marked CHARACTERIZATION pin behavior that is known to be wrong and is
scheduled to change. They exist to fire on a bump. When one goes red, convert it
to assert the corrected behavior -- do not delete it.
"""

import arf
import h5py as h5
import numpy as np
import pytest
from packaging.version import Version

# ------------------------------------------------------------------ constants


def test_spec_version():
    # Written into the arf_version attribute of every file arfx creates, and
    # compared against the ceiling in check_file_version.
    assert arf.spec_version == "2.2"


def test_library_version_is_parseable():
    assert Version(arf.__version__) >= Version("3.0.0")


def test_version_info_is_string():
    # core.arfx and oephys.script pass this straight to argparse's version action
    assert isinstance(arf.version_info(), str)


@pytest.mark.parametrize(
    "name,value",
    [
        ("UNDEFINED", 0),
        ("ACOUSTIC", 1),
        ("EXTRAC_HP", 2),
        ("EXTRAC_LF", 3),
        ("EXTRAC_EEG", 4),
        ("EVENT", 1000),
        ("SPIKET", 1001),
        ("BEHAVET", 1002),
        ("INTERVAL", 2000),
    ],
)
def test_datatype_codes(name, value):
    # These codes are persisted in the datatype attribute of stored datasets, so
    # a renumbering silently reinterprets every file already on disk.
    assert getattr(arf.DataTypes, name).value == value


def test_datatype_lookup_by_value():
    # core.entry_repr and core.ParseDataType both round-trip through this
    assert arf.DataTypes(1).name == "ACOUSTIC"
    with pytest.raises(ValueError):
        arf.DataTypes(999999)


# ------------------------------------------------------------------- fixtures


@pytest.fixture
def arf_file(tmp_path):
    """An arf file with one entry holding each of the dataset shapes arfx handles."""
    path = tmp_path / "probe.arf"
    with arf.open_file(path, "w") as fp:
        entry = arf.create_entry(fp, "entry", 1000)
        arf.create_dataset(
            entry, "sampled_1d", np.arange(1000, dtype="f"), sampling_rate=100
        )
        arf.create_dataset(
            entry, "sampled_2d", np.zeros((1000, 3), dtype="h"), sampling_rate=100
        )
        arf.create_dataset(
            entry, "event_seconds", np.array([0.5, 1.5, 2.5, 9.0]), units="s"
        )
        arf.create_dataset(
            entry,
            "event_samples",
            np.array([50, 150, 250, 900]),
            units="samples",
            sampling_rate=100,
        )
        arf.create_dataset(
            entry,
            "marked",
            np.rec.fromrecords(
                [(50, b"a"), (150, b"b"), (950, b"c")], names=("start", "name")
            ),
            units=(b"samples", b""),
            sampling_rate=100,
        )
        yield fp


# ------------------------------------------------------------------ open_file


def test_open_file_sets_root_attributes(tmp_path):
    with arf.open_file(tmp_path / "new.arf", "w") as fp:
        assert fp.attrs["arf_version"] == arf.spec_version
        assert fp.attrs["arf_library"] == "python"
        assert fp.attrs["arf_library_version"] == arf.__version__


def test_open_file_tracks_link_creation_order(tmp_path):
    # arfx relies on this for stable entry ordering; without it keys_by_creation
    # silently degrades to alphabetical (see test_keys_by_creation_untracked).
    path = tmp_path / "order.arf"
    with arf.open_file(path, "w") as fp:
        for name in ("zebra", "alpha", "middle"):
            arf.create_entry(fp, name, 1000)
    with arf.open_file(path, "r") as fp:
        assert list(arf.keys_by_creation(fp)) == ["zebra", "alpha", "middle"]


def test_open_file_does_not_clobber_existing_attributes(tmp_path):
    path = tmp_path / "reopen.arf"
    with arf.open_file(path, "w") as fp:
        fp.attrs["arf_library_version"] = "0.0.0"
    with arf.open_file(path, "a") as fp:
        assert fp.attrs["arf_library_version"] == "0.0.0"


# ---------------------------------------------------------------- create_entry


def test_create_entry_returns_group(arf_file):
    entry = arf_file["entry"]
    assert isinstance(entry, h5.Group)
    assert arf.is_entry(entry)


def test_create_entry_sets_timestamp_and_uuid(arf_file):
    attrs = arf_file["entry"].attrs
    assert set(attrs.keys()) >= {"timestamp", "uuid"}
    assert attrs["timestamp"].dtype == np.int64
    assert attrs["timestamp"].shape == (2,)
    assert attrs["uuid"].dtype == np.dtype("S36")


def test_create_entry_accepts_extra_attributes(tmp_path):
    with arf.open_file(tmp_path / "t.arf", "w") as fp:
        entry = arf.create_entry(fp, "e", 1000, experimenter="smm3rc", pen=1)
        assert entry.attrs["experimenter"] == "smm3rc"
        assert entry.attrs["pen"] == 1


def test_create_entry_rejects_name_with_slash(tmp_path):
    # A slash used to create a nested group rather than an entry at the root,
    # silently. arfx builds entry names from input filenames (core.iter_entries
    # uses Path.stem) and from -n templates whose fields come from HDF5
    # attributes, so a slash is reachable from user data; oephys.py guards
    # against it explicitly. The arfx-level consequence is covered by
    # test_core.py::test_add_entries_rejects_template_with_slash.
    with arf.open_file(tmp_path / "t.arf", "w") as fp:
        with pytest.raises(ValueError):
            arf.create_entry(fp, "a/b", 1000)
        assert "a" not in fp


# -------------------------------------------------------------- create_dataset


@pytest.mark.parametrize(
    "kwargs,message",
    [
        (dict(data=np.zeros(10)), "requires sampling_rate"),
        (dict(data=np.zeros(10), units="samples"), "requires sampling_rate"),
        (
            dict(data=np.rec.fromrecords([(1, 2)], names=("a", "b")), units=(b"", b"")),
            "requires 'start' field",
        ),
        (
            dict(
                data=np.rec.fromrecords([(1.0, b"x")], names=("start", "n")), units="s"
            ),
            "requires sequence of units",
        ),
        (
            dict(
                data=np.rec.fromrecords([(1.0, b"x")], names=("start", "n")),
                units=(b"s",),
            ),
            "number of units doesn't match",
        ),
    ],
)
def test_create_dataset_validation(tmp_path, kwargs, message):
    with arf.open_file(tmp_path / "t.arf", "w") as fp:
        entry = arf.create_entry(fp, "e", 1000)
        with pytest.raises(ValueError, match=message):
            arf.create_dataset(entry, "bad", **kwargs)


def test_create_dataset_rejects_string_arrays(tmp_path):
    # A numpy string array used to bypass the type check, which sits behind a
    # `not hasattr(data, "dtype")` guard, and fail later with a confusing message
    # about sampling_rate. Fixed in 2.7.3.
    with arf.open_file(tmp_path / "t.arf", "w") as fp:
        entry = arf.create_entry(fp, "e", 1000)
        with pytest.raises(ValueError, match="numeric or compound type"):
            arf.create_dataset(entry, "bad", np.array(["x", "y"]))


def test_create_dataset_sets_units_and_datatype(arf_file):
    dset = arf_file["entry"]["sampled_1d"]
    assert dset.attrs["units"] == ""
    assert dset.attrs["datatype"] == arf.DataTypes.UNDEFINED


def _as_text(units):
    return [u.decode("ascii") if isinstance(u, bytes) else str(u) for u in units]


@pytest.mark.parametrize(
    "units",
    [
        (b"samples", b""),
        [b"samples", b""],
        ["samples", ""],
        np.array([b"samples", b""]),
        np.array(["samples", ""]),
    ],
)
def test_create_dataset_accepts_any_units_sequence(tmp_path, units):
    # 3.0.0 tests the units argument structurally instead of requiring a list or
    # tuple. This is what select.py's read-modify-write needs: h5py hands the
    # attribute back as an ndarray, so the isinstance check meant a compound
    # dataset could not be copied from one file to another without converting
    # first. The unicode-dtype case is the one numpy produces from a list of
    # str, and h5py refuses it directly with "No conversion path for dtype".
    data = np.rec.fromrecords([(50, b"a")], names=("start", "name"))
    with arf.open_file(tmp_path / "t.arf", "w") as fp:
        entry = arf.create_entry(fp, "e", 1000)
        dset = arf.create_dataset(entry, "marked", data, units=units, sampling_rate=100)
        assert _as_text(dset.attrs["units"]) == ["samples", ""]


def test_units_storage_class_follows_the_input_form(tmp_path):
    # The value survives every input form, but the HDF5 type does not, and that
    # is visible to a reader. h5py stores a list or tuple as variable-length
    # strings, which come back as str; a fixed-width bytes array stays fixed
    # width and comes back as np.bytes_. So passing a units attribute read from
    # one file straight into another preserves its class, while rebuilding it as
    # a list -- which select.py does when it has to pad -- converts it. Anything
    # comparing this attribute has to decode first; arf._decode_attribute and
    # arfx's own predicates already do.
    data = np.rec.fromrecords([(50, b"a")], names=("start", "name"))
    with arf.open_file(tmp_path / "t.arf", "w") as fp:
        entry = arf.create_entry(fp, "e", 1000)
        from_list = arf.create_dataset(
            entry, "a", data, units=[b"samples", b""], sampling_rate=100
        )
        from_array = arf.create_dataset(
            entry, "b", data, units=np.array([b"samples", b""]), sampling_rate=100
        )
        assert from_list.attrs["units"].dtype == np.dtype("O")
        assert from_array.attrs["units"].dtype.kind == "S"


def test_create_dataset_rejects_units_sequence_on_time_series(tmp_path):
    # Only a compound dataset takes one unit per field. Rejecting this early
    # also keeps a sequence away from the `units == ""` comparison below it,
    # which on an ndarray is elementwise and raises the ambiguous-truth-value
    # ValueError from inside numpy rather than saying what is wrong.
    with arf.open_file(tmp_path / "t.arf", "w") as fp:
        entry = arf.create_entry(fp, "e", 1000)
        with pytest.raises(ValueError, match="only complex event data"):
            arf.create_dataset(
                entry, "d", np.zeros(10, dtype="h"), units=["mV", "mV"], sampling_rate=1
            )


def test_create_dataset_honors_compression_and_maxshape(tmp_path):
    # core.add_entries passes both through from the -z / -u flags
    with arf.open_file(tmp_path / "t.arf", "w") as fp:
        entry = arf.create_entry(fp, "e", 1000)
        dset = arf.create_dataset(
            entry,
            "d",
            np.zeros(1000, dtype="h"),
            sampling_rate=100,
            compression=9,
            maxshape=(None,),
        )
        assert dset.compression == "gzip"
        assert dset.compression_opts == 9
        assert dset.maxshape == (None,)


# ------------------------------------------------------------- set_attributes


def test_set_attributes_overwrite_default(arf_file):
    entry = arf_file["entry"]
    arf.set_attributes(entry, marker="first")
    arf.set_attributes(entry, marker="second")
    assert entry.attrs["marker"] == "second"


def test_set_attributes_no_overwrite(arf_file):
    # core.add_entries uses overwrite=False to stamp file_creator only once
    entry = arf_file["entry"]
    arf.set_attributes(entry, marker="first")
    arf.set_attributes(entry, marker="second", overwrite=False)
    assert entry.attrs["marker"] == "first"


def test_set_attributes_none_deletes(arf_file):
    entry = arf_file["entry"]
    arf.set_attributes(entry, marker="value")
    arf.set_attributes(entry, marker=None)
    assert "marker" not in entry.attrs


def test_set_attributes_none_on_missing_key_is_noop(arf_file):
    arf.set_attributes(arf_file["entry"], never_existed=None)


# ------------------------------------------------------------ keys_by_creation


def test_keys_by_creation_returns_str_keys(arf_file):
    keys = list(arf.keys_by_creation(arf_file["entry"]))
    assert all(isinstance(k, str) for k in keys)
    assert keys == [
        "sampled_1d",
        "sampled_2d",
        "event_seconds",
        "event_samples",
        "marked",
    ]


def test_keys_by_creation_untracked(tmp_path):
    # A file created by something other than arf.open_file has no creation-order
    # index. arf falls back to a visit shim rather than raising, so the order
    # silently becomes alphabetical -- arfx cannot distinguish the two cases.
    path = tmp_path / "plain.arf"
    with h5.File(path, "w") as fp:
        for name in ("zebra", "alpha", "middle"):
            fp.create_group(name)
        assert list(arf.keys_by_creation(fp)) == ["alpha", "middle", "zebra"]


# --------------------------------------------------------------- predicates


@pytest.mark.parametrize(
    "name,is_ts,is_mp,nchannels",
    [
        ("sampled_1d", True, False, 1),
        ("sampled_2d", True, False, 3),
        ("event_seconds", False, False, 1),
        ("event_samples", False, False, 1),
        ("marked", False, True, 1),
    ],
)
def test_dataset_predicates(arf_file, name, is_ts, is_mp, nchannels):
    dset = arf_file["entry"][name]
    assert arf.is_time_series(dset) is is_ts
    assert arf.is_marked_pointproc(dset) is is_mp
    assert arf.count_channels(dset) == nchannels


def test_is_entry_discriminates_groups_from_datasets(arf_file):
    assert arf.is_entry(arf_file["entry"])
    assert not arf.is_entry(arf_file["entry"]["sampled_1d"])


def test_is_entry_on_file_root(arf_file):
    # The root used to pass, because an h5py.File is a Group. The spec says it
    # is not an entry: a file may hold top-level datasets belonging to no entry.
    # arfx was unaffected by the change because select.py only ever asks about
    # children (select.py:69 filters keys_by_creation, select.py:153 walks
    # src.values()). Any future walk that starts at the root is not.
    assert arf.is_entry(arf_file) is False


def test_is_time_series_tolerates_bytes_units(tmp_path):
    # Files written by the C++ implementation store units as fixed-length
    # strings, which h5py returns as bytes. Before 2.7.3 the `units not in
    # ("s", "samples")` test was a str comparison, so b"samples" never matched
    # and a spike train read back as sampled data. collect.py filters on this
    # predicate, so the misclassification would have reached arfx output.
    with arf.open_file(tmp_path / "t.arf", "w") as fp:
        entry = arf.create_entry(fp, "e", 1000)
        dset = arf.create_dataset(
            entry, "spikes", np.array([1, 2]), units="samples", sampling_rate=100
        )
        dset.attrs["units"] = np.bytes_(b"samples")
        assert arf.is_time_series(dset) is False


def test_count_children(arf_file):
    assert arf.count_children(arf_file) == 1
    assert arf.count_children(arf_file, h5.Group) == 1
    assert arf.count_children(arf_file["entry"], h5.Dataset) == 5


# ------------------------------------------------------------ select_interval


def test_select_interval_returns_pair(arf_file):
    result = arf.select_interval(arf_file["entry"]["sampled_1d"], 1.0, 3.0)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_select_interval_time_series(arf_file):
    data, offset = arf.select_interval(arf_file["entry"]["sampled_1d"], 1.0, 3.0)
    # sampling_rate converts the bounds to samples before slicing
    assert offset == 100
    assert data.shape == (200,)
    assert data[0] == 100.0


def test_select_interval_marked_pointproc(arf_file):
    data, offset = arf.select_interval(arf_file["entry"]["marked"], 1.0, 3.0)
    assert offset == 100
    assert data["start"].tolist() == [50]  # 150 rebased to interval start
    assert data["name"].tolist() == [b"b"]


def test_select_interval_event_with_sampling_rate(arf_file):
    data, offset = arf.select_interval(arf_file["entry"]["event_samples"], 1.0, 3.0)
    assert offset == 100
    assert data.tolist() == [50, 150]


def test_select_interval_event_without_sampling_rate(arf_file):
    data, offset = arf.select_interval(arf_file["entry"]["event_seconds"], 1.0, 3.0)
    assert offset == 1.0
    assert data.tolist() == [0.5, 1.5]


def test_select_interval_seconds_units_are_not_rescaled(arf_file):
    # The unit of the returned offset follows the `units` attribute, NOT the mere
    # presence of sampling_rate. Before 2.7.3 this keyed on presence, so a
    # seconds-valued point process that also carried a sampling rate had its
    # [0, 1) second window silently reinterpreted as [0, 1000) samples.
    #
    # This matters to select.py, which adds the returned offset to the stored
    # offset attribute: under the old behavior it mixed seconds into a
    # samples-valued attribute for exactly these datasets.
    entry = arf_file["entry"]
    dset = arf.create_dataset(
        entry, "seconds_pp", np.array([0.5, 5.0]), units="s", sampling_rate=1000
    )
    data, offset = arf.select_interval(dset, 0.0, 1.0)
    assert offset == 0.0
    assert data.tolist() == [0.5]


def test_select_interval_tolerates_scalar_units_on_a_compound_dataset(tmp_path):
    # The spec wants one unit per field, but older versions of jrecord write a
    # single scalar describing the whole record, and examples/ has such a file.
    # 3.0.0 treats a scalar as applying to every field rather than indexing into
    # the string, which would have tested its first character against "samples"
    # and so read a sample-timebase spike train as if it were in seconds.
    # select.py has to build the file with plain h5py: create_dataset enforces
    # the per-field rule, which is exactly why the non-conforming files exist.
    data = np.rec.fromrecords([(50, b"a"), (150, b"b")], names=("start", "name"))
    with arf.open_file(tmp_path / "t.arf", "w") as fp:
        entry = arf.create_entry(fp, "e", 1000)
        dset = entry.create_dataset("marked", data=data)
        dset.attrs["units"] = b"samples"
        dset.attrs["sampling_rate"] = 100
        selected, offset = arf.select_interval(dset, 1.0, 3.0)
        # rescaled to samples, so the window is [100, 300)
        assert offset == 100
        assert selected["start"].tolist() == [50]


def test_select_interval_rebases_an_integer_start_field(tmp_path):
    # The subtraction is cast to the field's own type. begin is only an integer
    # when the window was rescaled to samples, and an in-place subtraction of a
    # float from an integer field raises rather than converting.
    data = np.rec.fromrecords([(250, b"a")], names=("start", "name"))
    with arf.open_file(tmp_path / "t.arf", "w") as fp:
        entry = arf.create_entry(fp, "e", 1000)
        dset = arf.create_dataset(
            entry, "m", data, units=(b"samples", b""), sampling_rate=100
        )
        selected, _offset = arf.select_interval(dset, 1.0, 3.0)
        assert selected["start"].dtype == data["start"].dtype
        assert selected["start"].tolist() == [150]


def test_select_interval_empty_dataset_preserves_dtype(arf_file):
    # Before 2.7.3 the `if idx.size > 0` guard tested the boolean mask rather
    # than the number of matches, so an empty dataset returned the mask itself --
    # a bool array where callers expect the dataset's dtype.
    entry = arf_file["entry"]
    dset = arf.create_dataset(entry, "empty", np.array([], dtype="f"), units="s")
    data, _ = arf.select_interval(dset, 0.0, 1.0)
    assert data.dtype == np.dtype("f")
    assert data.size == 0


def test_select_interval_is_half_open(arf_file):
    # [begin, end) -- the sample at end is excluded
    data, _ = arf.select_interval(arf_file["entry"]["event_samples"], 0.0, 1.5)
    assert data.tolist() == [50]


# -------------------------------------------------------- check_file_version


def _versioned_file(path, **attrs):
    with h5.File(path, "w") as fp:
        for key, value in attrs.items():
            fp.attrs[key] = value
    return path


def test_check_file_version_returns_version(tmp_path):
    path = _versioned_file(tmp_path / "ok.arf", arf_version="2.1")
    with h5.File(path, "r") as fp:
        result = arf.check_file_version(fp)
    assert isinstance(result, Version)
    assert result == Version("2.1")


def test_check_file_version_accepts_bytes_attribute(tmp_path):
    path = _versioned_file(tmp_path / "b.arf", arf_version=np.bytes_(b"2.1"))
    with h5.File(path, "r") as fp:
        assert arf.check_file_version(fp) == Version("2.1")


def test_check_file_version_falls_back_to_library_version(tmp_path):
    # A legacy accommodation for files written before arf_version existed. Only
    # ever safe while the library and spec versions shared a scale.
    path = _versioned_file(tmp_path / "lib.arf", arf_library_version="2.1")
    with h5.File(path, "r") as fp:
        assert arf.check_file_version(fp) == Version("2.1")


def test_check_file_version_refuses_modern_library_version_as_fallback(tmp_path):
    # The other half of the fallback, narrowed in 3.0.0: the scales diverged, so
    # a library version at or above the spec ceiling no longer stands in for a
    # spec version. This is the case a file written by arf 3.x with no
    # arf_version would hit -- which the library does not do, but the C++
    # implementation and older third-party writers are not bound by that.
    path = _versioned_file(tmp_path / "lib3.arf", arf_library_version="3.0.0")
    with h5.File(path, "r") as fp:
        with pytest.raises(UserWarning):
            arf.check_file_version(fp)


@pytest.mark.parametrize(
    "attrs,exc",
    [
        ({}, UserWarning),
        ({"arf_version": "not-a-version"}, UserWarning),
        ({"arf_version": "1.0"}, DeprecationWarning),
        ({"arf_version": "3.0"}, FutureWarning),
    ],
)
def test_check_file_version_raises(tmp_path, attrs, exc):
    # NB these are raised as exceptions, not emitted via the warnings module.
    # Every arfx call site funnels through core.check_file_version, which catches
    # `Warning` and logs, so the shared base class is what makes the check
    # advisory rather than fatal.
    #
    # The unparseable case previously escaped as packaging's InvalidVersion,
    # which is not a Warning subclass and so passed straight through that
    # handler. Fixed in 2.7.3.
    path = _versioned_file(tmp_path / "bad.arf", **attrs)
    with h5.File(path, "r") as fp:
        with pytest.raises(exc):
            arf.check_file_version(fp)


def test_supported_spec_versions_brackets_the_implemented_spec():
    # The range check_file_version enforces, reported rather than inferred. The
    # ceiling is derived: the next major after the implemented spec, on the
    # reasoning that a minor revision cannot change a required attribute.
    low, high = arf.supported_spec_versions()
    assert low == arf.min_spec_version
    assert Version(low) <= Version(arf.spec_version) < Version(high)
    assert Version(high).major == Version(arf.spec_version).major + 1
    assert Version(high).minor == 0


def test_check_file_version_accepts_just_under_the_ceiling(tmp_path):
    # Brackets the ceiling from below. Built from the reported range so it keeps
    # testing the boundary rather than the number 2.9 after the spec moves.
    high = Version(arf.supported_spec_versions()[1])
    under = f"{high.major - 1}.999"
    path = _versioned_file(tmp_path / "edge.arf", arf_version=under)
    with h5.File(path, "r") as fp:
        assert arf.check_file_version(fp) == Version(under)


def test_check_file_version_floor_is_the_reported_minimum(tmp_path):
    # The floor rose from 1.1 to 2.0 in arf 3.0.0: the required attributes
    # changed at spec 2.0, so the library will not vouch for anything older.
    # arfx has no migration path for such files -- the python 2 migrate module
    # was removed in 2.8.1 -- so all it can do is report them.
    low = Version(arf.supported_spec_versions()[0])
    path = _versioned_file(tmp_path / "old.arf", arf_version=f"{low.major - 1}.5")
    with h5.File(path, "r") as fp:
        with pytest.raises(DeprecationWarning):
            arf.check_file_version(fp)


# ----------------------------------------------------------------- file_version


def test_file_version_reports_without_judging(tmp_path):
    # New in 3.0.0, and the counterpart to the test above: a caller whose job is
    # handling old files has to read the version *because* it is out of range.
    # arfx has no such caller now that migrate is gone, but this is the hook a
    # future one would use.
    low = Version(arf.supported_spec_versions()[0])
    old = f"{low.major - 1}.5"
    path = _versioned_file(tmp_path / "old.arf", arf_version=old)
    with h5.File(path, "r") as fp:
        assert arf.file_version(fp) == Version(old)


def test_file_version_still_raises_on_unreadable_version(tmp_path):
    # The one thing it cannot do is report a version that is not there.
    path = _versioned_file(tmp_path / "none.arf")
    with h5.File(path, "r") as fp:
        with pytest.raises(UserWarning):
            arf.file_version(fp)


# ------------------------------------------------------- check_file_structure


def test_check_file_structure_passes_a_conforming_file(arf_file):
    assert arf.check_file_structure(arf_file) == []


def test_check_file_structure_flags_a_shared_dataset(tmp_path):
    # A dataset hard-linked into two entries has no defined time, since an
    # entry's timestamp is what places its datasets. Nothing prevents this at
    # write time -- it takes plain h5py to do, not the arf API -- which is why
    # the check exists separately.
    with arf.open_file(tmp_path / "t.arf", "w") as fp:
        one = arf.create_entry(fp, "one", 1000)
        two = arf.create_entry(fp, "two", 2000)
        arf.create_dataset(one, "d", np.zeros(10, dtype="h"), sampling_rate=100)
        two["d"] = one["d"]
        problems = arf.check_file_structure(fp)
    # NB one defect, two problems: the walk reports the dataset from each entry
    # it is reachable through, so the length of the list is not a defect count.
    assert problems == [
        "dataset 'one/d' is linked into more than one entry",
        "dataset 'two/d' is linked into more than one entry",
    ]


def test_check_file_structure_ignores_top_level_datasets(tmp_path):
    # The spec allows datasets outside any entry -- select.py copies them
    # through verbatim -- so they must not be reported.
    with arf.open_file(tmp_path / "t.arf", "w") as fp:
        fp.create_dataset("log", data=np.zeros(4, dtype="h"))
        arf.create_entry(fp, "one", 1000)
        assert arf.check_file_structure(fp) == []


# ------------------------------------------------------------------ timestamps


def test_convert_timestamp_from_float():
    ts = arf.convert_timestamp(1234567890.5)
    assert ts.dtype == np.int64
    assert ts.shape == (2,)
    assert ts.tolist() == [1234567890, 500000]


def test_convert_timestamp_from_int():
    assert arf.convert_timestamp(1234567890).tolist() == [1234567890, 0]


def test_convert_timestamp_from_datetime():
    import datetime

    dt = datetime.datetime(2023, 10, 16, 16, 30, 54, 250000)
    ts = arf.convert_timestamp(dt)
    assert arf.timestamp_to_datetime(ts) == dt


def test_convert_timestamp_from_pair():
    assert arf.convert_timestamp((100, 250)).tolist() == [100, 250]


def test_convert_timestamp_honors_tzinfo():
    # Before 2.7.3 this went through mktime(obj.timetuple()), which reads the
    # wall-clock fields as local time and discards the offset, recording an
    # aware datetime as the wrong instant by the local zone offset.
    import datetime

    aware = datetime.datetime(2023, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)
    assert arf.convert_timestamp(aware).tolist() == [int(aware.timestamp()), 0]


def test_convert_timestamp_negative_float_borrows():
    # The spec defines the second element as the rest of the elapsed time, which
    # cannot be negative. -1.5 is (-2 seconds, +500000 us), not (-1, -500000).
    assert arf.convert_timestamp(-1.5).tolist() == [-2, 500000]
    assert arf.timestamp_to_float(arf.convert_timestamp(-1.5)) == -1.5


def test_convert_timestamp_rejects_garbage():
    with pytest.raises(TypeError):
        arf.convert_timestamp(object())


def test_timestamp_round_trip_through_entry(tmp_path):
    # splitter.py and core.py both read the stored attribute back out
    with arf.open_file(tmp_path / "t.arf", "w") as fp:
        entry = arf.create_entry(fp, "e", 1234567890.5)
        assert arf.timestamp_to_float(entry.attrs["timestamp"]) == 1234567890.5


def test_timestamp_to_float():
    assert arf.timestamp_to_float(np.array([100, 500000])) == 100.5
