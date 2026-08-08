# -*- mode: python -*-
"""
Code for moving data in and out of arf containers.  There are some
function entry points for performing common tasks, and several script
entry points.

Functions
=====================
add_entries:      add entries from various containers to an arf file
extract_entries:  extract entries from arf file to various containers
delete_entries:   delete entries from arf file
list_entries:     generate a list of all the entries/channels in a file

Scripts
=====================
arfx:      general-purpose compression/extraction utility with tar-like syntax
"""

import argparse
import logging
import os
import shutil
import subprocess
from collections.abc import Container, Iterable, Sequence
from functools import lru_cache
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

import arf
import h5py as h5

from . import __version__, io

# template for extracted files
default_extract_template = "{entry}_{channel}.wav"
# template for created entries
default_entry_template = "{base}_{index:04}"
log = logging.getLogger("arfx")  # root logger


def check_file_version(arfp: h5.File) -> None:
    """Check a file's ARF version, logging a warning if it may be incompatible.

    arf signals incompatibility by raising the warning classes rather than
    emitting them through the warnings module, and the check is advisory on
    both sides, so every operation logs and carries on. Half the call sites
    used to abort instead, which meant the same file was readable or not
    depending on which subcommand you used.
    """
    try:
        arf.check_file_version(arfp)
    except DeprecationWarning as e:
        # arf's message for this case offers to upgrade the file with a script
        # from this package, which was true until the python 2 migrate module
        # was removed in 2.8.1. Say what is actually on offer instead.
        log.warning("warning: %s", e)
        log.warning(
            "         arfx cannot convert files older than ARF spec %s; "
            "reading it anyway, but expect missing attributes",
            arf.supported_spec_versions()[0],
        )
    except Warning as e:
        log.warning("warning: %s", e)


def entry_repr(entry: h5.Group) -> str:
    from h5py import h5t

    attrs = entry.attrs
    out = str(entry.name)
    for k, v in attrs.items():
        if k.isupper():
            continue
        if k == "timestamp":
            ts = arf.timestamp_to_datetime(v).strftime("%Y-%m-%d %H:%M:%S.%f")
            out += f"\n  timestamp : {ts}"
        else:
            out += f"\n  {k} : {v}"
    for name, dset in entry.items():
        out += f"\n  /{name} :"
        if isinstance(dset.id.get_type(), h5t.TypeVlenID):
            out += " vlarray"
        else:
            out += f" array {dset.shape}"
            if "sampling_rate" in dset.attrs:
                out += f" @ {dset.attrs['sampling_rate']:.1f}/s"
            if dset.dtype.names is not None:
                out += " (compound type)"

        units = dset.attrs.get("units", "")
        try:
            units = units.decode("ascii")
        except AttributeError:
            pass
        out += f", units '{units}'"
        try:
            datatype_value = dset.attrs["datatype"]
            datatype_name = arf.DataTypes(datatype_value).name
        except KeyError:
            datatype_name = arf.DataTypes.UNDEFINED.name
        except ValueError:
            datatype_name = f"UNKNOWN ({datatype_value})"
        out += f", type {datatype_name}"
        if dset.compression:
            if dset.compression_opts is not None:
                out += f" [{dset.compression}{dset.compression_opts}]"
            else:
                out += f" [{dset.compression}]"
    return out


def dataset_properties(dset: h5.Dataset) -> tuple[str, str, int]:
    """Infers the type of data and some properties of an hdf5 dataset.

    Returns tuple: (sampled|event|interval|unknown), (array|table|vlarry), ncol
    """
    from h5py import h5t

    interval_dtype_names = ("name", "start", "stop")
    dtype = dset.id.get_type()

    if isinstance(dtype, h5t.TypeVlenID):
        return "event", "vlarray", dset.id.shape[0]

    if isinstance(dtype, h5t.TypeCompoundID):
        # table types; do a check on the dtype for backwards compat with 1.0
        names, ncol = dtype.dtype.names, dtype.get_nmembers()
        if "start" not in names:
            contents = "unknown"
        elif any(k not in names for k in interval_dtype_names):
            contents = "event"
        else:
            contents = "interval"
        return contents, "table", ncol

    dtt = dset.attrs.get("datatype", 0)
    ncols = (len(dset.shape) < 2 and 1) or dset.shape[1]
    if dtt < arf.DataTypes.EVENT:
        # assume UNKNOWN is sampled
        return "sampled", "array", ncols
    else:
        return "event", "array", ncols


def pluralize(n: int, sing: str = "", plurl: str = "s") -> str:
    """Returns 'sing' if n == 1, else 'plurl'"""
    if n == 1:
        return sing
    else:
        return plurl


def parse_name_template(
    node: h5.Group | h5.Dataset,
    template: str,
    index: int = 0,
    default: str = "NA",
) -> str:
    """Generates names for output files using a template and the entry/dataset attributes

    see http://docs.python.org/library/string.html#format-specification-mini-language for template formatting

    node - a dataset or group object
    template - string with formatting codes, e.g. {animal}
               Values are looked up in the dataset attributes, and then the parent entry attributes.
               (entry) and (channel) refer to the name of the entry and dataset
    index - value to insert for {index} key (usually the index of the entry in the file)
    default - value to replace missing keys with
    """
    from string import Formatter

    f = Formatter()
    values = dict()
    entry = dset = None
    if isinstance(node, h5.Group):
        entry = node
    elif isinstance(node, h5.Dataset):
        dset = node
        entry = dset.parent

    try:
        for _lt, field, _fs, _c in f.parse(template):
            if field is None:
                continue
            elif field == "entry":
                if not entry:
                    raise ValueError(f"can't resolve `entry` field for {node}")
                values[field] = PurePosixPath(entry.name).name
            elif field == "channel":
                if not dset:
                    raise ValueError(f"can't resolve `channel` field for {node}")
                values[field] = PurePosixPath(dset.name).name
            elif field == "index":
                values[field] = index
            elif dset is not None and hasattr(dset, field):
                values[field] = getattr(dset, field)
            elif dset is not None and field in dset.attrs:
                values[field] = dset.attrs[field]
            elif entry is not None and hasattr(entry, field):
                values[field] = getattr(entry, field)
            elif entry is not None and field in entry.attrs:
                values[field] = entry.attrs[field]
            else:
                values[field] = default
        if values:
            return f.format(template, **values)
        else:
            return template  # no substitutions were made
    except ValueError as e:
        raise ValueError(f"template error: {e.message}") from e


def iter_entries(src: Path | str, cbase: str = "pcm"):
    """Iterate through the entries and channels of a data source.

    Yields (data, entry index, entry name,)
    """
    src = Path(src)
    fp = io.open(src, "r")
    fbase = src.stem
    nentries = getattr(fp, "nentries", 1)
    for entry in range(nentries):
        try:
            fp.entry = entry
        except AttributeError:
            pass

        if nentries == 1:
            yield fp, entry, fbase
        else:
            ename = default_entry_template.format(base=fbase, index=entry)
            yield fp, entry, ename


def add_entries(
    tgt: Path | str,
    files: Sequence[Path | str],
    *,
    compress: str | None = None,
    template: str | None = None,
    datatype: arf.DataTypes | None = arf.DataTypes.UNDEFINED,
    attrs: dict | None = None,
    **options,
) -> None:
    """
    Add data to a file. This is a general-purpose function that will
    iterate through the entries in the source files (or groups of
    files) and add the data to the target file.  The source data can
    be in any file format understood by io.open.
    """
    chan = "pcm"  # only pcm data can be imported

    if len(files) == 0:
        raise FileNotFoundError("must specify one or more input files")

    if attrs is None:
        attrs = {}
    with arf.open_file(tgt, "a") as arfp:
        check_file_version(arfp)
        arf.set_attributes(
            arfp, file_creator="org.meliza.arfx/arfx " + __version__, overwrite=False
        )
        next_entry_index = arf.count_children(arfp, h5.Group)
        for f in files:
            for fp, entry_index, entry_name in iter_entries(f):
                timestamp = getattr(fp, "timestamp", None)
                if timestamp is None:
                    # kludge for ewave
                    if hasattr(fp, "fp") and hasattr(fp.fp, "fileno"):
                        timestamp = os.fstat(fp.fp.fileno()).st_mtime
                    else:
                        raise ValueError(
                            f"{f}/{entry_index} missing required timestamp"
                        )
                if not hasattr(fp, "sampling_rate"):
                    raise ValueError(
                        f"{f}/{entry_index} missing required sampling_rate attribute"
                    )

                if template is not None:
                    entry_name = default_entry_template.format(
                        base=template, index=next_entry_index
                    )
                entry = arf.create_entry(
                    arfp,
                    entry_name,
                    timestamp,
                    entry_creator="org.meliza.arfx/arfx " + __version__,
                    **attrs,
                )
                arf.create_dataset(
                    entry,
                    chan,
                    fp.read(),
                    datatype=datatype,
                    sampling_rate=fp.sampling_rate,
                    compression=compress,
                    source_file=str(f),
                    source_entry=entry_index,
                )
                next_entry_index += 1
                log.debug("%s/%d -> /%s/%s", f, entry_index, entry_name, chan)


def create_and_add_entries(
    tgt: Path | str, files: Sequence[Path | str], **options
) -> None:
    """Add data to a new file. If the file exists it's deleted"""
    tgt = Path(tgt)
    if tgt.is_file():
        tgt.unlink()
    add_entries(tgt, files, **options)


def extract_entries(
    src: Path | str,
    entries: Container[str] | None = None,
    *,
    directory: Path | str | None = None,
    channels: Container[str] | None = None,
    template: str | None = None,
    **options,
):
    """
    Extract entries from a file.  The format and naming of the output
    containers is determined automatically from the name of the entry
    and the type of data.

    entries: list of the entries to extract. Can be None, in which
             case all the entries are extracted
    channels: list of the channels to extract. Can be None, in which
              case all of the channels are extracted.
    template: if specified, name the output files sequentially
    """
    src = Path(src)
    if not src.is_file():
        raise FileNotFoundError(f"the file {src} does not exist")
    directory = Path(".") if directory is None else Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"the target directory {directory} does not exist")

    with arf.open_file(src, "r") as arfp:
        check_file_version(arfp)
        for index, ename in enumerate(arfp):
            entry = arfp[ename]
            attrs = dict(entry.attrs)
            # entries written by other tools may have no timestamp; leave the
            # output file's mtime alone rather than passing None to os.utime
            timestamp = attrs.get("timestamp", None)
            mtime = timestamp[0] if timestamp is not None else None
            if entries is None or ename in entries:
                for channel in entry:
                    if channels is not None and channel not in channels:
                        log.debug("%s -> skipped (not requested)", channel)
                        continue
                    dset = entry[channel]
                    attrs.update(
                        nchannels=dset.shape[1] if len(dset.shape) > 1 else 1,
                        dtype=dset.dtype,
                        **dset.attrs,
                    )
                    fname = directory / parse_name_template(
                        dset, template or default_extract_template, index=index
                    )
                    dtype, _stype, _ncols = dataset_properties(dset)
                    if dtype != "sampled":
                        log.debug("%s -> skipped (no supported containers)", dset.name)
                        continue

                    with io.open(fname, "w", **attrs) as fp:
                        fp.write(dset)
                    if mtime is not None:
                        os.utime(fname, (os.stat(fname).st_atime, mtime))

                    log.debug("%s -> %s", dset.name, fname)


def delete_entries(
    src: Path | str, entries: Iterable[str], *, repack: bool = False, **options
):
    """
    Delete one or more entries from a file.

    entries: list of the entries to delete
    repack: if True, repack the file afterward to reclaim space
    """
    src = Path(src)
    if not src.is_file():
        raise FileNotFoundError(f"the file {src} does not exist")

    count = 0
    with arf.open_file(src, "r+") as arfp:
        check_file_version(arfp)
        for entry in entries:
            if entry in arfp:
                try:
                    del arfp[entry]
                    count += 1
                    log.debug("deleted /%s", entry)
                except Exception as e:
                    log.error("unable to delete %s: %s", entry, e)
            else:
                log.debug("unable to delete %s: no such entry", entry)
    if count > 0 and repack:
        repack_file(src, **options)


def copy_entries(
    tgt: Path | str,
    files: Iterable[Path | str],
    entry_base: str | None = None,
    **options,
) -> None:
    """
    Copy data from another arf file. Arguments can refer to entire arf
    files (just the filename) or specific entries (using path
    notation).  Record IDs and all other metadata are copied with the entry.

    entry_base: if specified, rename entries sequentially in target file using this base
    """
    acache = lru_cache(maxsize=None)(arf.open_file)
    with arf.open_file(tgt, "a") as arfp:
        check_file_version(arfp)
        for f in files:
            # this is a bit tricky:
            # file.arf is a file; file.arf/entry is entry
            # dir/file.arf is a file; dir/file.arf/entry is entry
            # on windows, dir\file.arf/entry is an entry
            src = Path(f)
            if src.is_file():
                items = ((src, entry) for _, entry in acache(f, mode="r").items())
            elif src.parent.is_file():
                fp = acache(src.parent, mode="r")
                try:
                    items = ((src.parent, fp[src.name]),)
                except KeyError:
                    log.error("unable to copy %s: no such entry", f)
                    continue
            else:
                log.error("unable to copy %s: does not exist", f)
                continue

            n_entries_in_target = arf.count_children(arfp, h5.Group)
            for i, (fname, entry) in enumerate(items, start=n_entries_in_target):
                if entry_base is not None:
                    entry_name = default_entry_template.format(base=entry_base, index=i)
                else:
                    entry_name = PurePosixPath(entry.name).name
                arfp.copy(entry, arfp, name=entry_name)
                log.debug("%s%s -> %s/%s", fname, entry.name, tgt, entry_name)


def list_entries(
    src: Path | str, entries: Iterable[str] | None = None, **options
) -> None:
    """
    List the contents of the file, optionally restricted to specific entries

    entries: if None, list all entries; otherwise only list entries
             that are in this list (more verbosely)
    """
    print(f"{src}:")
    with arf.open_file(src, "r") as arfp:
        check_file_version(arfp)
        if entries is None:
            for name in arfp:
                entry = arfp[name]
                if isinstance(entry, h5.Dataset):
                    print(f"{entry.name}: top-level dataset")
                elif options.get("verbose", False):
                    print(entry_repr(entry))
                else:
                    n_channels = len(entry)
                    print(f"{entry.name}: {n_channels} channel{pluralize(n_channels)}")
        else:
            for ename in entries:
                if ename in arfp:
                    print(entry_repr(arfp[ename]))


def update_entries(
    src: Path | str,
    entries: Container[str] | None,
    *,
    verbose: bool = False,
    **metadata,
):
    """
    Update metadata on one or more entries.

    entries: List of entries to update. If None, updates all entries.
    metadata: key-value pairs to set (old values are kept)
    """
    src = Path(src)
    if not src.is_file():
        raise FileNotFoundError(f"the file {src} does not exist")

    with arf.open_file(src, "r+") as arfp:
        check_file_version(arfp)
        for entry_name in arfp:
            entry_name = PurePosixPath(entry_name).name
            if entries is None or entry_name in entries:
                enode = arfp[entry_name]
                if verbose:
                    print("vvvvvvvvvv")
                    print(entry_repr(enode))
                    print("**********")
                arf.set_attributes(enode, **metadata)
                if verbose:
                    print(entry_repr(enode))
                    print("^^^^^^^^^^")


def write_toplevel_attribute(
    tgt: Path | str, files: Iterable[Path | str], **options
) -> None:
    """Store contents of files as text in top-level attribute with basename of each file"""
    with arf.open_file(tgt, "a") as arfp:
        check_file_version(arfp)
        for fname in files:
            fname = Path(fname)
            attrname = f"user_{fname.name}"
            print(f"{fname} -> {tgt}/{attrname}")
            arfp.attrs[attrname] = fname.read_text()


def read_toplevel_attribute(
    src: Path | str, attrnames: Iterable[str], **options
) -> None:
    """Print text data stored in top-level attributes by write_toplevel_attribute()"""
    with arf.open_file(src, "r") as arfp:
        check_file_version(arfp)
        for attrname in attrnames:
            aname = f"user_{attrname}"
            print(f"{aname}:")
            try:
                print(arfp.attrs[aname])
            except KeyError:
                print(" - no such attribute")


def repack_file(src: Path | str, *, compress: int | None = None, **options) -> None:
    """Call h5repack on a file"""
    src_path = Path(src)
    if not src_path.is_file():
        raise FileNotFoundError(f"Source file not found: {src_path}")
    cmd = ["/usr/bin/env", "h5repack"]
    if compress is not None:
        cmd.extend(("-f", "SHUF", "-f", f"GZIP={compress:d}"))
    with TemporaryDirectory() as temp_dir:
        tgt_file = Path(temp_dir) / src_path.name
        try:
            result = subprocess.run(
                [*cmd, str(src_path), str(tgt_file)],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                log.info("Repacked %s", src_path)
                shutil.copy2(tgt_file, src_path)
            else:
                log.error(
                    "Failed to repack %s, keeping original. Error: %s",
                    src_path,
                    result.stderr.strip(),
                )
        except subprocess.SubprocessError as e:
            log.exception("Error executing h5repack command: %s", e)


class ParseKeyVal(argparse.Action):
    """Collect KEY=VALUE arguments, reading each value as JSON where it parses.

    HDF5 attributes are typed, and this used to store every value as a string,
    so `-k pen=1` could not be read back as a number and there was no way to say
    otherwise from the command line. JSON is the rule because it supplies
    numbers, booleans and lists without a syntax of its own, and because its
    quoting doubles as the escape hatch: `-k animal='"397"'` keeps a
    numeric-looking identifier a string.

    A value that is not valid JSON is taken as a bare string, which is what most
    metadata is -- including anything with a hyphen or a leading zero, so dates
    and padded identifiers are safe.
    """

    def __call__(self, parser, namespace, arg, option_string=None):
        import json

        kv = getattr(namespace, self.dest) or dict()
        key, sep, val = arg.partition("=")
        # partition rather than split: a value may itself contain '='
        if not sep or not key:
            parser.error(f"{option_string} {arg} is badly formed; needs key=value")
        try:
            value = json.loads(val)
        except ValueError:
            value = val
        if isinstance(value, dict):
            parser.error(
                f"{option_string} {key}: an HDF5 attribute cannot hold a mapping"
            )
        kv[key] = value
        setattr(namespace, self.dest, kv)


class ParseDataType(argparse.Action):
    def __call__(self, parser, namespace, arg, option_string=None):
        if arg.isdigit():
            setattr(namespace, self.dest, arf.DataTypes(int(arg)))
        else:
            setattr(namespace, self.dest, arf.DataTypes[arg])


def setup_log(log, debug=False):
    ch = logging.StreamHandler()
    formatter = logging.Formatter("[%(name)s] %(message)s")
    loglevel = logging.DEBUG if debug else logging.INFO
    log.setLevel(loglevel)
    ch.setLevel(loglevel)
    ch.setFormatter(formatter)
    log.addHandler(ch)


def datatype_list():
    out = str(arf.DataTypes.__doc__)
    for dtype in arf.DataTypes:
        out += f"\n{dtype.name}:{dtype.value}"
    return out


def format_list():
    fmts = io.list_plugins()
    return f"Supported file formats: {' '.join(fmts)}"


def arfx(argv=None):
    p = argparse.ArgumentParser(
        description="copy data in and out of ARF files",
    )
    p.add_argument("entries", nargs="*")
    p.add_argument("--version", action="version", version="%(prog)s " + __version__)
    p.add_argument("--arf-version", action="version", version=arf.version_info())
    p.add_argument(
        "--help-datatypes",
        help="print available datatypes and exit",
        action="version",
        version=datatype_list(),
    )
    p.add_argument(
        "--help-formats",
        help="list supported file types and exit",
        action="version",
        version=format_list(),
    )
    # operations
    pp = p.add_argument_group("Operations")
    g = pp.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "-A",
        help="copy data from another ARF file",
        action="store_const",
        dest="op",
        const=copy_entries,
    )
    g.add_argument(
        "-c",
        help="create new file and add data",
        action="store_const",
        dest="op",
        const=create_and_add_entries,
    )
    g.add_argument(
        "-r",
        help="add data to an existing file",
        action="store_const",
        dest="op",
        const=add_entries,
    )
    g.add_argument(
        "-x",
        help="extract one or more entries from the ARF file",
        action="store_const",
        dest="op",
        const=extract_entries,
    )
    g.add_argument(
        "-t",
        help="list contents of the file",
        action="store_const",
        dest="op",
        const=list_entries,
    )
    g.add_argument(
        "-U",
        help="update metadata of entries",
        action="store_const",
        dest="op",
        const=update_entries,
    )
    g.add_argument(
        "-d",
        help="delete entries",
        action="store_const",
        dest="op",
        const=delete_entries,
    )
    g.add_argument(
        "--write-attr",
        help="add text file(s) to top-level attribute(s)",
        action="store_const",
        dest="op",
        const=write_toplevel_attribute,
    )
    g.add_argument(
        "--read-attr",
        help="read top-level attribute(s)",
        action="store_const",
        dest="op",
        const=read_toplevel_attribute,
    )

    g = p.add_argument_group("Options")
    g.add_argument(
        "-f",
        help="the ARF file to operate on",
        required=True,
        metavar="FILE",
        dest="arffile",
    )
    g.add_argument("-v", help="verbose output", action="store_true", dest="verbose")
    g.add_argument(
        "-n",
        help="name destination entries or files using python format strings. "
        "Replacement fields include {entry}, {channel}, and {index} as well as any "
        "attribute of the dataset or parent entry. Example: `{animal}_{index:04}.wav`",
        metavar="TEMPLATE",
        dest="template",
    )
    g.add_argument(
        "-C",
        help="during extraction, include this channel (default all)",
        metavar="CHANNEL",
        dest="channels",
        nargs="+",
    )
    g.add_argument(
        "-T",
        help="specify data type (see --help-datatypes)",
        default=arf.DataTypes.UNDEFINED,
        metavar="DATATYPE",
        dest="datatype",
        action=ParseDataType,
    )
    g.add_argument(
        "-k",
        help=(
            "specify attributes of entries; the value is read as JSON when it parses (so pen=1 is a number), otherwise as a string"
        ),
        action=ParseKeyVal,
        metavar="KEY=VALUE",
        dest="attrs",
    )
    g.add_argument(
        "-P",
        help="repack when deleting entries",
        action="store_true",
        dest="repack",
    )
    g.add_argument(
        "-z",
        help="set compression level in ARF (default: %(default)s)",
        type=int,
        default=1,
        dest="compress",
    )
    g.add_argument(
        "--directory",
        help="when extracting files, store them in this directory",
        type=Path,
    )

    args = p.parse_args(argv)
    setup_log(log, args.verbose)

    try:
        opts = args.__dict__.copy()
        entries = opts.pop("entries") or None
        args.op(args.arffile, entries, **opts)
    except (ValueError, FileNotFoundError) as err:
        # ValueError also covers arf's rejection of an entry name containing a
        # path separator, which -n templates can produce from file names and
        # HDF5 attributes
        p.error(f"[arfx] error: {err}")


if __name__ == "__main__":
    arfx()

# Variables:
# End:
