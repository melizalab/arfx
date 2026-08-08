# Audit findings and the arf 3.0 migration

Working notes from the 2.8.1 audit. Everything fixed in that release is
described in `NEWS.md`; this file records what was deliberately left alone and
what the next major version has to deal with.

## Deferred from 2.8.1 — all resolved in 3.0.0

These changed observable behavior, which is why they were kept out of a patch
release. `NEWS.md` describes what each one became. Two are worth keeping a
record of here, because what was found was not what was written down.

- **The `extract_entries` / `list_entries` ordering divergence does not
  exist.** Measured both ways: on a file written by `arf.open_file`, direct
  iteration and `keys_by_creation` both give creation order, because h5py
  honours the tracking flag; on an untracked file both give name order,
  because HDF5 falls back to the name index. They agree in both cases, and
  `list_entries` was iterating directly too, so the note was wrong about who
  used what as well.

  Chasing it did turn up two real bugs. `extract_entries` and `update_entries`
  treated top-level datasets as entries, and every jrecord file has one:
  `arfx -x` raised a `TypeError` from inside h5py after walking the log
  table's rows as channel names, and `arfx -U` wrote entry metadata onto the
  table. Both now go through `core.entries_by_creation`.

- **`io.py`'s dead code was hiding a live bug.** Removing the unreachable
  python < 3.10 shims meant looking at the `try` around them, which wrapped the
  handler's construction as well as the entry-point lookup — so a `ValueError`
  a handler raised for its own reasons came back as "No handler defined for
  files of type '.dat'", pointing away from the actual problem.

The lesson both times: verify the finding before fixing it. A stale note is
worth as much as an accurate one only if someone checks.

## Suggested upstream, in arf

- ~~**`create_dataset` accepts only `list` or `tuple` for a compound dataset's
  `units`.**~~ Fixed in arf 3.0.0, which tests the argument structurally. Half
  of `select.py`'s workaround went away with it.

- **`check_file_version`'s `DeprecationWarning` says "The arfx package ships a
  script that upgrades old files".** It has not since arfx 2.8.1, when the
  python 2 `migrate` module was removed, and arfx prints that sentence to its
  own users. `core.check_file_version` appends a correction, but the message
  wants fixing at the source.

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
