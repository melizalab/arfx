# Release notes

## 2.8.1

A bug-fix release. Two of the scripts did not work on their primary inputs, and
one flag did the opposite of what it documented, affecting `arfx-select`,
`arfx-oephys` and `arfx-collect-sampled`. However, most of the affected
behaviors were not used in the lab, so the impact is expected to be minimal.

Requires `arf>=2.7.4`, which carries fixes this release depends on.

### arfx-select

- Could not process a spec-conforming marked point process at all. The `units`
  attribute was removed from the dataset's attributes and restored only in the
  branch that repairs jrecord's non-conforming scalar, so a conforming
  attribute was lost and the write was rejected for having no units. Separately,
  h5py returns the attribute as an array while `arf.create_dataset` requires a
  list or tuple, so even a preserved attribute was refused. Both are fixed;
  marked point processes now round-trip whether or not their units conform.
- Log messages were attributed to `arfx-collect` rather than `arfx-select`.

### arfx-oephys

- The sample-offset fallback did not work with recordings from GUI 0.6 or
  later. When the sample index file is missing, the recording start is
  recovered from `sync_messages.txt`; that path still expected the pre-0.6
  wording and a `source_processor_sub_idx` key that 0.6 removed, so it raised
  `KeyError` from inside its own exception handler. Both wordings are now
  recognised, and a stream is matched by name on newer files.
- An aborted recording, which leaves a zero-length `continuous.dat` beside a
  zero-length sample index, raised `IndexError` while the recording was being
  opened — before `--skip-empty` could skip it. An empty index is now treated
  like a missing one.

### arfx-collect-sampled

- `-e/--entries` excluded the entries it named instead of restricting output to
  them.
- Any file whose entries were not all the same length was rejected as
  inconsistent, because the consistency check compared each dataset's HDF5
  chunk size, which h5py derives from its length. A session holding more than
  one recording was the common casualty. Chunk size no longer takes part in the
  comparison.
- An inconsistent file now exits non-zero with the diagnostic that had already
  been logged, instead of raising `TypeError` over the top of it.

### npy files

- Reading or writing a `.npy` file raised `AttributeError` under numpy 2.5.
  `npyio` obtained the npy v1.0 header layout from
  `numpy.lib.format._header_size_info`, a private mapping numpy 2.5 removed
  with no public replacement. The layout is fixed by the format specification,
  so it is now written out directly. arfx declares no numpy constraint, so any
  fresh install picks up numpy 2.5 and hit this.
- Two internal `array.shape = ...` assignments, deprecated in numpy 2.5, are
  now `reshape` calls. No behavior change; both were already views.

### arfx-split

- No longer writes a trailing entry containing zero samples when the recording
  length divides evenly by the chunk duration. The chunk count is now derived
  per dataset in samples, so rounding in a seconds-based division cannot add a
  spurious chunk either. An entry with no sampled data still produces one
  entry, so event-only entries keep their attributes.
- A `--duration` shorter than one sample is rejected with a message rather than
  producing a run of empty entries.

### arfx

- `-x` no longer raises `TypeError` when extracting an entry that has no
  `timestamp` attribute; the output file keeps its own modification time.
- Every operation now logs a warning and continues when a file's ARF version
  may be incompatible. Four operations did that already and five aborted
  instead, so whether a file was readable depended on which operation you used.
  The version check is advisory in the arf library, and arfx now treats it that
  way consistently.

### Removed

- `arfx/migrate.py`. It was Python 2 code: it imported `distutils`, removed in
  Python 3.12, and a module (`arfx.h5vlen`) that does not exist in the package,
  so it could not be imported on any supported interpreter. It had no console
  script entry point either. Recoverable from version control if support for
  pre-2.0 ARF files is ever needed, but it would need rewriting rather than
  repairing.

### Packaging

- `h5py>=3.15` and `packaging` are now declared dependencies. Both were always
  direct imports — h5py in three modules, `packaging.version` in `oephys.py` —
  and resolved only because `arf` happened to require them. The h5py floor is
  the first release with musllinux/aarch64 wheels.

### Documentation

- The `-P` flag was documented backwards: it opts in to repacking after a
  delete, rather than suppressing it.
