# -*- coding: utf-8 -*-
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
import sys
import shutil
import subprocess
from collections.abc import Container, Iterable, Sequence
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Optional, Union
from tempfile import TemporaryDirectory

import arf
import h5py as h5

from . import __version__, io

# template for extracted files
default_extract_template = "{entry}_{channel}.wav"
# template for created entries
default_entry_template = "{base}_{index:04}"
log = logging.getLogger("arfx")  # root logger


def entry_repr(entry: h5.Group) -> str:
    from h5py import h5t

    attrs = entry.attrs
    out = str(entry.name)
    for k, v in attrs.items():
        if k.isupper():
            continue
        if k == "timestamp":
            out += "\n  timestamp : %s" % arf.timestamp_to_datetime(v).strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )
        else:
            out += f"\n  {k} : {v}" % v
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
    node: Union[h5.Group, h5.Dataset],
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


def iter_entries(src: Union[Path, str], cbase: str = "pcm"):
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
    tgt: Union[Path, str],
    files: Sequence[Union[Path, str]],
    *,
    compress: Optional[str] = None,
    template: Optional[str] = None,
    datatype: Optional[arf.DataTypes] = arf.DataTypes.UNDEFINED,
    attrs: Optional[dict] = None,
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
        raise ValueError("must specify one or more input files")

    if attrs is None:
        attrs = {}
    with arf.open_file(tgt, "a") as arfp:
        arf.check_file_version(arfp)
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
    tgt: Union[Path, str], files: Sequence[Union[Path, str]], **options
) -> None:
    """Add data to a new file. If the file exists it's deleted"""
    tgt = Path(tgt)
    if tgt.is_file():
        tgt.unlink()
    add_entries(tgt, files, **options)


def extract_entries(
    src: Union[Path, str],
    tgt_dir: Union[Path, str, None],
    *,
    entries: Optional[Container[str]] = None,
    channels: Optional[Container[str]] = None,
    template: Optional[str] = None,
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
    tgt_dir = Path(".") if tgt_dir is None else Path(tgt_dir)
    if not tgt_dir.is_dir():
        raise FileNotFoundError(f"the target directory {tgt_dir} does not exist")

    with arf.open_file(src, "r") as arfp:
        try:
            arf.check_file_version(arfp)
        except Warning as e:
            log.warning("warning: %s", e)
        for index, ename in enumerate(arfp):
            entry = arfp[ename]
            attrs = dict(entry.attrs)
            mtime = attrs.get("timestamp", [None])[0]
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
                    fname = tgt_dir / parse_name_template(
                        dset, template or default_extract_template, index=index
                    )
                    dtype, _stype, _ncols = dataset_properties(dset)
                    if dtype != "sampled":
                        log.debug("%s -> skipped (no supported containers)", dset.name)
                        continue

                    with io.open(fname, "w", **attrs) as fp:
                        fp.write(dset)
                    os.utime(fname, (os.stat(fname).st_atime, mtime))

                    log.debug("%s -> %s", dset.name, fname)


def delete_entries(src: Union[Path, str], entries: Iterable[str], *, repack: bool = True, **options):
    """
    Delete one or more entries from a file.

    entries: list of the entries to delete
    repack: if True (default), repack the file afterward to reclaim space
    """
    src = Path(src)
    if not src.is_file():
        raise FileNotFoundError(f"the file {src} does not exist")

    count = 0
    with arf.open_file(src, "r+") as arfp:
        arf.check_file_version(arfp)
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
    tgt: Union[Path, str],
    files: Iterable[Union[Path, str]],
    entry_base: Optional[str] = None,
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
        arf.check_file_version(arfp)
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
    src: Union[Path, str], entries: Optional[Iterable[str]] = None, **options
) -> None:
    """
    List the contents of the file, optionally restricted to specific entries

    entries: if None, list all entries; otherwise only list entries
             that are in this list (more verbosely)
    """
    if not os.path.exists(src):
        raise FileNotFoundError(f"the file {src} does not exist")
    print(f"{src}:")
    with arf.open_file(src, "r") as arfp:
        try:
            arf.check_file_version(arfp)
        except Warning as e:
            log.warning("warning: %s", e)
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


def update_entries(src: Union[Path, str], entries: Optional[Container[str]], *, verbose: bool = False, **metadata):
    """
    Update metadata on one or more entries.

    entries: List of entries to update. If None, updates all entries.
    metadata: key-value pairs to set (old values are kept)
    """
    src = Path(src)
    if not src.is_file():
        raise FileNotFoundError(f"the file {src} does not exist")

    with arf.open_file(src, "r+") as arfp:
        try:
            arf.check_file_version(arfp)
        except Warning as e:
            log.warning("warning: %s", e)
        for i, entry_name in enumerate(arfp):
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


def write_toplevel_attribute(tgt: Union[Path, str], files: Iterable[Union[Path, str]], **options) -> None:
    """Store contents of files as text in top-level attribute with basename of each file"""
    with arf.open_file(tgt, "a") as arfp:
        try:
            arf.check_file_version(arfp)
        except Warning as e:
            log.warning("warning: %s", e)
        for fname in files:
            fname = Path(fname)
            attrname = f"user_{fname.name}"
            print(f"{fname} -> {tgt}/{attrname}")
            arfp.attrs[attrname] = fname.read_text()


def read_toplevel_attribute(src: Union[Path, str], attrnames: Iterable[str], **options) -> None:
    """Print text data stored in top-level attributes by write_toplevel_attribute()"""
    with arf.open_file(src, "r") as arfp:
        try:
            arf.check_file_version(arfp)
        except Warning as e:
            log.warning("warning: %s", e)
        for attrname in attrnames:
            aname = f"user_{attrname}"
            print(f"{aname}:")
            try:
                print(arfp.attrs[aname])
            except KeyError:
                print(" - no such attribute")


def repack_file(src: Union[Path, str], *, compress: Optional[int] = None, **options) -> None:
    """Call h5repack on a file"""
    src_path = Path(src)
    if not src_path.is_file():
        raise FileNotFoundError(f"Source file not found: {src_path}")
    cmd = ["/usr/bin/env", "h5repack"]
    if compress is not None:
        cmd.extend(("-f", "SHUF", "-f", "GZIP=%d" % compress))
    with TemporaryDirectory() as temp_dir:
        tgt_file = Path(temp_dir) / src_path.name
        try:
            result = subprocess.run(
                [*cmd, str(src_path), str(tgt_file)],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                log.info("Repacked %s", src_path)
                shutil.copy2(tgt_file, src_path)
            else:
                log.error("Failed to repack %s, keeping original. Error: %s", src_path, result.stderr.strip())
        except subprocess.SubprocessError as e:
            log.exception("Error executing h5repack command: %s", e)
            

class ParseKeyVal(argparse.Action):
    def __call__(self, parser, namespace, arg, option_string=None):
        kv = getattr(namespace, self.dest)
        if kv is None:
            kv = dict()
        if not arg.count("=") == 1:
            raise ValueError("-k %s argument badly formed; needs key=value" % arg)
        else:
            key, val = arg.split("=")
            kv[key] = val
        setattr(namespace, self.dest, kv)


# class ParseDataType(argparse.Action):
#     def __call__(self, parser, namespace, arg, option_string=None):
#         try:
#             if arg.isdigit():
#                 argx = 
#         if not arg.isdigit():
#             argx = arf.DataTypes._fromstring(arg)
#             if argx is None:
#                 raise ValueError("'%s' is not a valid data type" % arg)
#         else:
#             argx = arg
#         setattr(namespace, self.dest, int(argx))


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


def arfx():
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
        type=arf.DataTypes,
        default=arf.DataTypes.UNDEFINED,
        metavar="DATATYPE",
        dest="datatype",
        #action=ParseDataType,
    )
    g.add_argument(
        "-k",
        help="specify attributes of entries",
        action=ParseKeyVal,
        metavar="KEY=VALUE",
        dest="attrs",
    )
    g.add_argument(
        "-P",
        help="don't repack when deleting entries",
        action="store_false",
        dest="repack",
    )
    g.add_argument(
        "-z",
        help="set compression level in ARF (default: %(default)s)",
        type=int,
        default=1,
        dest="compress",
    )

    args = p.parse_args()
    setup_log(log, args.verbose)

    try:
        opts = args.__dict__.copy()
        entries = opts.pop("entries")
        args.op(args.arffile, entries, **opts)
    except Exception as e:
        print("[arfx] error: %s" % e)
        if isinstance(e, DeprecationWarning):
            print("      use arfx-migrate to convert to version %s" % arf.spec_version)
        raise e
        sys.exit(-1)
    return 0


if __name__ == "__main__":
    sys.exit(arfx())

# Variables:
# End:
