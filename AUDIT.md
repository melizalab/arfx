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

## The arf 3.0 bump, as it actually went

The boundary suite in `test/test_arf_api.py` existed for this, and it did its
job: running it first and alone produced the API diff instead of a guess.

**Five failures, all of them predicted.** Four were `CHARACTERIZATION` tests
written to fire on this bump, and one was the recorded `spec_version` constant.
The full suite produced the same five failures and no others, so no arfx
behavior changed — this was a dependency-floor release, not a behavior release.

| What changed | Why arfx was unaffected |
|---|---|
| `is_entry` excludes the file root | `select.py` only asks about children, at lines 69 and 153 |
| `create_entry` rejects `/` in a name | reachable only from a `-n` template; now a CLI error rather than a silently nested group |
| supported spec range is `[2.0, 3.0)`, derived | every call site funnels through `core.check_file_version`, which logs |
| `arf_library_version` fallback refused at ≥3.0 | arfx never writes a file without `arf_version` |
| `spec_version` is 2.2 | only ever compared, never parsed for meaning |

Three things surfaced from reading the diff rather than from a test:

- `core.arfx()`'s `except DeprecationWarning` handler had become unreachable in
  2.8.1, when every call site moved behind the logging wrapper. Removed.
- arf's `DeprecationWarning` text tells the user that "the arfx package ships a
  script that upgrades old files". It does not, since 2.8.1. `core.check_file_version`
  now appends a correction; **the message itself should be fixed upstream in arf.**
- `select.py`'s units workaround was half redundant. arf 3.0 tests the argument
  structurally, so a conforming attribute is now passed through untouched, which
  preserves its HDF5 type. Only the padding path still rebuilds it as a list.

The bump also brought API arfx does not use yet, now pinned in the boundary
suite so the *next* bump is also a diff: `file_version`, `check_file_structure`,
`supported_spec_versions`, and `min_spec_version`.

### Procedure, for the next one

1. `uv add "arf>=N"`, then run `test/test_arf_api.py` **first and alone**.
   Whatever fails there is the API diff; read it as the changelog.
2. Run the full suite. Failures elsewhere that are not explained by a failure in
   step 1 are arfx integration bugs, not arf changes.
3. Convert each `CHARACTERIZATION` test that went red to assert the corrected
   behavior. Do not delete them.
4. Pin whatever new API the release added, while its behavior is fresh.

### Versioning

`README.rst` used to say arfx was "synchronized with the major/minor version
numbers of the arf package specification". That rule stopped meaning anything
when arf decoupled its library version from the spec version: arf 3.0.0 ships
against spec 2.2. Following the spec would have meant arfx never left 2.x.

arfx 3.0.0 therefore tracks the arf **library** major version it requires, and
the dependency is capped at `<4` so the next one cannot arrive silently — a
major bump is exactly when the boundary suite needs running.
