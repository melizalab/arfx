# Audit findings and the arf 3.0 migration

Working notes from the 2.8.1 audit. Everything fixed in that release is
described in `NEWS.md`; this file records what was deliberately left alone and
what the next major version has to deal with.

## Deferred to the next major version

All of these change observable behavior, which is why they were kept out of a
patch release. Each is reachable from real input unless noted.

### Ordering and naming

- **`oephys` entry names embed the absolute source path.** The name is
  `str(dir).replace("/", "_")`, so importing the same recording from two
  machines, or from a different mount point, produces different entry names for
  identical data. Something derived from the recording directory and the
  experiment/recording indices would be reproducible. Changing it renames
  entries in new archives, so it needs a deliberate release.

- **`extract_entries` iterates the file directly while `list_entries` and
  `collect` use `keys_by_creation`.** For files written by `arf.open_file` the
  two agree, because creation order is tracked and h5py honours it. They
  diverge for files written by anything else, where `keys_by_creation` silently
  falls back to alphabetical. The `{index}` field in a `-n` template therefore
  means different things depending on both the operation and the writer.

### Robustness

- **`core.ParseKeyVal` does no type coercion.** `-k pen=1` is stored and read
  back as the string `"1"`. Numeric metadata cannot be round-tripped through
  the command line. Fixing it changes the dtype of attributes in new files.

- **`select.py` calls `arf.create_entry(tgt, name, **src_entry_attrs)`.** This
  satisfies the positional `timestamp` parameter only because `timestamp`
  happens to be among the source attributes, and it feeds an already-converted
  int64 pair back through `convert_timestamp`. An entry with no timestamp
  raises `TypeError` on a missing argument.

- **`collect.first` returns `None` and `collect.all_items_equal` returns `True`
  for an empty dict.** If no channel matches the filter, the sampling rate
  becomes `None` and is passed to the output writer rather than being reported
  as "no channels selected".

- **`collect`'s `--start` and `--stop` are applied per chunk, not per sample.**
  The cutoff is the first chunk boundary at or past the requested count, so the
  output is longer than asked for by up to one chunk. `--start` also uses
  `sample_count > args.start`, which drops the chunk containing the requested
  start.

### Dead code

- **`io.py`**: `_get_handler_class` is never called, and the `TypeError`
  "shim for python < 3.10" branches in `open` and `list_plugins` cannot be
  reached under `requires-python = ">=3.11"`.

## Suggested upstream, in arf

- **`create_dataset` accepts only `list` or `tuple` for a compound dataset's
  `units`.** h5py returns the attribute as an ndarray, so a units attribute
  cannot survive a read-modify-write round trip without an explicit conversion.
  `select.py` converts locally; accepting any sequence would match the tolerant
  reader principle applied elsewhere in arf.

## The arf 3.0 migration

The boundary suite in `test/test_arf_api.py` exists for this. It asserts the
contract of every `arf.*` symbol arfx uses, so the bump produces a diff rather
than a guess. Tests in it marked `CHARACTERIZATION` pin behavior known to be
changing, and are meant to fail.

### Procedure

1. `uv add "arf>=3.0.0"`, then run `test/test_arf_api.py` **first and alone**.
   Whatever fails there is the API diff; read it as the changelog.
2. Run the full suite. Failures elsewhere that are not explained by a failure in
   step 1 are arfx integration bugs, not arf changes.
3. Convert each `CHARACTERIZATION` test that went red to assert the corrected
   behavior. Do not delete them.
4. Apply the deferred items above, which is the point of the major version.

### Known changes to expect

From the arf side, three that arfx touches directly:

- **`is_entry` will return `False` for the file root.** An `h5py.File` is a
  `Group`, so the root currently passes. `select.py` only ever asks about
  children, so it is unaffected, but any new code that walks from the root is.
  Pinned by `test_is_entry_on_file_root`.

- **`create_entry` will reject a name containing `/`.** arfx builds entry names
  from input filenames (`core.iter_entries` uses `Path.stem`) and from `-n`
  templates whose fields come from HDF5 attributes, so a slash is reachable from
  user data. `oephys.py` already guards against it. Pinned by
  `test_create_entry_accepts_name_with_slash`.

- **The supported spec range becomes derived rather than hard-coded**, moving
  from `[1.1, 3.0)` to `[2.0, 3.0)`, and the `arf_library_version` fallback is
  refused for library versions at or above 3.0. Files at 1.x that are accepted
  today will start raising. arfx has no migration path for them at all now that
  the Python 2 `migrate` module is gone. Pinned by
  `test_check_file_version_floor_is_one_one` and
  `test_check_file_version_falls_back_to_library_version`.

`test_supported_spec_versions_not_yet_present` fires when arf gains
`supported_spec_versions()`; at that point the two boundary tests should derive
their expectations from it instead of hard-coding the range.

### The version-numbering question

`README.rst` says arfx "uses semantic versioning and is synchronized with the
major/minor version numbers of the arf package specification". That rule no
longer says what it used to: arf has deliberately decoupled its library version
from the spec version, so arf 3.0.0 ships against spec 2.1. Following the arf
*library* to 3.0 would be following the wrong number, and following the *spec*
would mean arfx never leaves 2.x.

The rule needs rewording before the next major release, and the choice of
number for it is a decision in its own right.
