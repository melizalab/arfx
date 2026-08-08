# Release notes

## 3.0.0

Requires `arf>=3.0.0,<4`. The major version is the dependency: arfx's major
version now tracks the arf **library** major version it requires.

The arf bump itself changed no arfx behavior — the boundary suite in
`test/test_arf_api.py` was written to produce a diff rather than a guess, and
it did: five failures, every one of them a test written in advance to fire on
this bump, and nothing else in the suite moved. What follows is the behavior
that was deliberately held back from 2.8.1 because it could not go in a patch
release.

### arfx

- `-k key=value` values are now typed. The value is read as JSON if it parses
  as JSON, and taken as a plain string if it does not, so `-k pen=1` stores a
  number where it used to store `"1"`. Numeric metadata could not previously
  be round-tripped through the command line at all. `-k bird=C194` and
  `-k date=2026-08-08` are unaffected, and neither is `-k box=007`, since JSON
  rejects a leading zero — a padded identifier stays a string. Quote a value to
  force it: `-k animal='"397"'`. A key and value now split on the first `=`, so
  a value may contain one, and a malformed argument is a CLI error rather than
  a traceback.
- `-x` and `-U` no longer treat a top-level dataset as an entry. The
  specification allows datasets outside any entry and jill writes a `jill_log`
  table at the root of every file, so this was reachable from any jrecord
  recording: `-x` raised `TypeError` from inside h5py after walking the log
  table's rows as channel names, and `-U` wrote entry metadata onto the table.
  `{index}` in a `-n` template now counts entries only.
- An entry name containing `/` is rejected with a CLI error instead of quietly
  creating a nested group. Reachable from a `-n` template.

### arfx-oephys

- Entry names are now derived from the recording rather than from the path to
  it. The name was the full source path with separators replaced, so the same
  recording imported from a different working directory, machine or mount point
  produced a different entry name for identical data, and two archives of one
  recording could not be merged without duplicates. The name is now the session
  directory plus the components below it. **This renames entries in new
  archives**; existing files are untouched.
- A recording that is already in the target file is logged and skipped instead
  of raising `ValueError` from h5py. This case only became detectable once the
  names were reproducible.
- `-n/--template` is removed. It was accepted and never read.

### arfx-split

- Entries that are contiguous on the sample timeline are now spliced back
  together before chunking (#43), so a recording the recorder broke into pieces
  — or that an earlier `arfx-split` run chunked — is re-chunked as one
  continuous stream instead of inheriting the old boundaries. Runs are assembled
  after entries are sorted by timestamp, so a recording continuing into another
  file splices too. `--no-splice` restores the previous behavior.
- Contiguity is decided from a frame counter on each entry, `jack_frame` by
  default and settable with `--frame-attr`; entries without it are never
  spliced. The counter is a uint32 that wraps about every 27 hours at 44.1 kHz
  — one of the reasons a long recording ends up split in the first place — and
  the comparison is done modulo that wrap.
- Entries that overlap by up to `--max-overlap` samples (default 4096) are
  spliced with the duplicates dropped. The overlapping samples are compared
  first; if the two entries disagree about the data they share, the frame
  counter and the data cannot both be right, so they are left unspliced and
  reported.
- The frame counter on each output entry now advances with the chunk. Every
  chunk used to inherit the source entry's value, so they all claimed to start
  at the same sample, and `arfx-split` output could not be spliced by a later
  run — half of what #43 asks for.

### arfx-collect-sampled

- `--start` and `--stop` now cut at the sample they name. They were applied a
  chunk at a time: `--start` compared the sample count *before* the current
  chunk, so the chunk containing the requested sample was dropped whole — with
  the default chunk size, `--start 100` discarded several thousand samples —
  and `--stop` ran to the first chunk boundary at or past the request. A chunk
  boundary is an artifact of how h5py stored the dataset and should not be
  visible in the output.
- An empty channel selection is now an error naming the channels that could not
  be found. It used to pass silently: the sampling rate and dtype reached the
  output writer as `None` and produced a file with no channels.

### arfx-select

- Each selected interval gets its own `uuid`, and the source entry's name is
  recorded in a `source_entry` attribute. The source attributes were passed
  straight to `arf.create_entry`, so every interval cut from an entry inherited
  that entry's identity — two intervals from one source entry produced two
  entries in the output carrying one uuid, which is the case the attribute
  exists to distinguish.
- An entry with no `timestamp` is logged and skipped rather than raising
  `TypeError` and killing the whole run.
- A conforming `units` attribute is now passed through unchanged, which
  preserves its HDF5 type. arf 3.0 accepts any sequence, so only the padding
  path for jrecord's non-conforming scalar still rebuilds it.

### Library

- `arfx.io.open` no longer disguises an error raised by a handler. The guard
  covered the handler's construction as well as the entry-point lookup, so
  `io.open("x.dat", mode="q")` reported `No handler defined for files of type
  '.dat'` — pointing away from the actual complaint, which was about the mode.
- `arfx.io.list_plugins` was annotated `-> str` while returning a list.
- Removed: `io._get_handler_class`, never called, and the "shim for python <
  3.10" branches in `io.open` and `io.list_plugins`, unreachable under
  `requires-python = ">=3.11"`.

### Packaging

- `arf>=3.0.0,<4`. The upper bound is deliberate: a major arf bump is exactly
  when `test/test_arf_api.py` needs running, so it should not arrive through a
  non-frozen install.

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
