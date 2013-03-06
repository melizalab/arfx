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

__version__ = "2.0.0"

import os, sys, getopt, posixpath
import arf
import io
from tools import filecache

# defaults for user options
defaults = {
    'verbose' : False,
    'entry_base' : None,
    'datatype' : arf.DataTypes.UNDEFINED,
    'compress' : 1,
    'repack' : True,
    'split_sites' : False,
    'push_db' : None,
    'entry_attrs' : {},
    }
# template for extracted files
default_extract_template = "{entry}_{channel}.wav"
# template for created entries
default_entry_template = "{base}_{index:04}"


def get_data_type(a):
    if a.isdigit():
        defaults['datatype'] = int(a)
    else:
        defaults['datatype'] = arf.DataTypes._fromstring(a)
        if defaults['datatype'] is None:
            print >> sys.stderr, "Error: %s is not a valid data type" % a
            print >> sys.stderr, arf.DataTypes._doc()
            sys.exit(-1)


def parse_name_template(dset, template, index=0, default="NA"):
    """ Generates names for output files using a template and the entry/dataset attributes

    see http://docs.python.org/library/string.html#format-specification-mini-language for template formatting

    dset - a dataset object
    template - string with formatting codes, e.g. {animal}
               Values are looked up in the dataset attributes, and then the parent entry attributes.
               (entry) and (channel) refer to the name of the entry and dataset
    index - value to insert for {index} key (usually the index of the entry in the file)
    default - value to replace missing keys with
    """
    import posixpath as pp
    from string import Formatter
    f = Formatter()
    values = dict()
    entry = dset.parent
    try:
        for lt,field,fs,c in f.parse(template):
            if field is None:
                continue
            elif field == "entry":
                values[field] = pp.basename(entry.name)
            elif field == "channel":
                values[field] = pp.basename(dset.name)
            elif field == "index":
                values[field] = index
            elif field in dset.attrs:
                values[field] = dset.attrs[field]
            elif field in entry.attrs:
                values[field] = entry.attrs[field]
            else:
                values[field] = default
        if values:
            return f.format(template, **values)
        else:
            return template  # no substitutions were made
    except ValueError, e:
        raise ValueError("Error in template: " + e.message)


def iter_entries(src, cbase='pcm'):
    """
    Iterate through the entries and channels of a data source.
    Yields (data, entry index, entry name,)
    """
    fp = io.open(src, 'r')
    fbase = os.path.splitext(os.path.basename(src))[0]
    nentries = getattr(fp,'nentries',1)
    for entry in xrange(nentries):
        try:
            fp.entry = entry
        except:
            pass

        if nentries==1:
            yield fp, entry, fbase
        else:
            ename = default_entry_template.format(base=fbase, index=entry)
            yield fp, entry, ename

def add_entries(tgt, files, **options):
    """
    Add data to a file. This is a general-purpose function that will
    iterate through the entries in the source files (or groups of
    files) and add the data to the target file.  The source data can
    be in any file format understood by io.open.

    Additional keyword arguments specify metadata on the newly created
    entries.
    """
    compress = options.get('compress',None)
    ebase = options.get('entry_base',None)
    chan = "pcm"                # only pcm data can be imported
    arfp = arf.file(tgt,'a')
    try:
        metadata = options['entry_attrs']
        for f in files:
            try:
                for fp,entry_index,entry_name in iter_entries(f):
                    timestamp = getattr(fp,'timestamp',None)
                    if timestamp is None:
                        # kludge for ewave
                        if hasattr(fp,'fp') and hasattr(fp.fp,'fileno'):
                            timestamp = os.fstat(fp.fp.fileno()).st_mtime
                        else:
                            raise ValueError, "File is missing required timestamp"
                    if not hasattr(fp,'sampling_rate'):
                        raise ValueError, "File is missing required sampling_rate attribute"

                    if ebase is not None:
                        entry_name = default_entry_template.format(base=ebase, index=arfp.nentries)
                    entry = arfp.create_entry(entry_name, timestamp, **metadata)

                    entry.add_data(chan, fp.read(),
                                   datatype=options['datatype'],
                                   sampling_rate=fp.sampling_rate,
                                   compression=compress,
                                   source_file=f,
                                   source_entry=entry_index)
                    if options['verbose']:
                        print "%s/%d -> /%s/%s" % (f, entry_index, entry_name, chan)
            except TypeError, e:
                print "%s: Unrecognized format (%s)" % (f,e)
            except IOError, e:
                print "%s: Error opening file (%s)" % (f,e)
            except ValueError, e:
                print "%s: Error creating entry (%s)" % (f,e)
    finally:
        arfp.__exit__(None,None,None)


def extract_entries(src, entries, **options):
    """
    Extract entries from a file.  The format and naming of the output
    containers is determined automatically from the name of the entry
    and the type of data.

    entries: list of the entries to extract. can be None, in which
             case all the entries are extracted
    entry_base: if specified, name the output files sequentially
    """
    if len(entries)==0: entries = None
    arfp = arf.file(src,'r')
    try:
        for index,(ename,entry) in enumerate(arfp.items(key='timestamp')):
            attrs = dict(entry.attrs)
            if entries is None or ename in entries:
                for channel in entry:
                    dset = entry[channel]
                    attrs.update(nchannels=dset.shape[1] if len(dset.shape)>1 else 1,
                                 dtype=dset.dtype,
                                 **dset.attrs)
                    fname = parse_name_template(dset,
                                                options.get('entry_base',None) or default_extract_template,
                                                index=index)
                    fp = io.open(fname, 'w', **attrs) # will throw error for unsupported format

                    dtype,stype,ncols = arf.dataset_properties(dset)
                    if dtype!='sampled': continue

                    fp.write(dset)
                    if 'timestamp' in entry.attrs:
                        fp.timestamp = entry.attrs['timestamp'][0]
                    else:
                        # try to set the modification time
                        pass

                    if options['verbose']:
                        print "%s -> %s" % (dset.name, fname)
    finally:
        arfp.__exit__(None,None,None)


def delete_entries(src, entries, **options):
    """
    Delete one or more entries from a file.

    entries: list of the entries to delete
    repack: if True (default), repack the file afterward to reclaim space
    """
    if entries is None or len(entries)==0: return
    arfp = arf.file(src,'r+')
    try:
        count = 0
        for entry in entries:
            if entry in arfp:
                try:
                    arfp.delete_entry(entry)
                    count += 1
                    if options['verbose']:
                        print "/%s" % entry
                except Exception, e:
                    print "Error deleting %s: %s" % (entry, e)
            elif options['verbose']:
                print "/%s: no such entry" % entry
    finally:
        arfp.__exit__(None,None,None)
    if count > 0 and options['repack']:
        repack_file((src,),**options)

def copy_entries(tgt, files, **options):
    """
    Copy data from another arf file. Arguments can refer to entire arf
    files (just the filename) or specific entries (using path
    notation).  Record IDs and all other metadata are copied with the entry.

    entry_base: if specified, rename entries sequentially in target file
    """
    ebase = options.get('entry_base',None)
    arfp = arf.file(tgt,'a')
    acache = filecache(arf.file)

    for f in files:
        # this is a bit tricky:
        # file.arf is a file; file.arf/entry is entry
        # dir/file.arf is a file; dir/file.arf/entry is entry
        # on windows, dir\file.arf/entry is an entry
        pn, fn = posixpath.split(f)
        if os.path.isfile(f):
            it = ((f,entry) for ename, entry in acache[f].items())
        elif os.path.isfile(pn):
            fp = acache[pn]
            if fn in fp:
                it = ((pn,fp[fn]),)
            else:
                print "Error: no such entry %s" % f
                continue
        else:
            print "Error: %s does not exist" % f
            continue

        for fname,entry in it:
            try:
                if ebase is not None: entry_name = "%s_%04d" % (ebase, arfp.nentries)
                else: entry_name=posixpath.basename(entry.name)
                arfp.h5.copy(entry, arfp.h5, name=entry_name)
                if options['verbose']:
                    print "%s%s -> %s/%s" % (fname, entry.name, tgt, entry_name)
            except ValueError:
                print "Error: can't create entry %s/%s; already exists?" % (tgt,entry_name)

    arfp.__exit__(None,None,None)
    acache.__exit__(None,None,None)


def list_entries(src, entries, **options):
    """
    List the contents of the file, optionally restricted to specific entries

    entries: if None or empty, list all entries; otherwise only list entries
             that are in this list (more verbosely)
    """
    arfp = arf.file(src,'r')
    print "%s:" % src
    try:
        if entries is None or len(entries)==0:
            for name,entry in arfp.items(key='timestamp'):
                if options.get('verbose',False):
                    print entry
                else:
                    print "%s: %d channel%s" % (entry.name, entry.nchannels,
                                                arf.pluralize(entry.nchannels))
        else:
            for ename in entries:
                if ename in arfp: print arfp[ename]
    finally:
        arfp.__exit__(None,None,None)

def update_entries(src, entries, **options):
    """
    Update metadata on one or more entries

    entries: if None or empty, updates all entries. In this case, if the
             name parameter is set, the entries are renamed sequentially
    """
    ebase = options.get('entry_base',None)
    if (entries is None or len(entries)==0) and ebase is not None and ebase.find('%') < 0:
        ebase += '_%04d'

    arfp = arf.file(src,'r+')
    try:
        for i,entry in enumerate(arfp):
            if entries is None or len(entries)==0 or posixpath.relpath(entry) in entries:
                enode = arfp[entry]
                if options.get('verbose',False):
                    print "vvvvvvvvvv"
                    print enode.__str__()
                    print "**********"
                if ebase:
                    name = ebase % i
                    arfp.h5[name] = enode
                    del arfp.h5[entry] # entry object should remain valid
                arfp.set_attributes(enode, **options['entry_attrs'])
                if options.get('verbose',False):
                    print enode.__str__()
                    print "^^^^^^^^^^"
    finally:
        arfp.__exit__(None,None,None)


def repack_file(files, **options):
    """ Call h5repack on a list of files to repack them """
    from shutil import rmtree, copy
    from tempfile import mkdtemp

    cmd = '/usr/bin/env h5repack '
    compress = options.get('compress',False)
    if compress:
        cmd += "-f SHUF -f GZIP=%d " % compress
    try:
        tdir = mkdtemp()
        for f in files:
            if options['verbose']:
                sys.stdout.write("Repacking %s..." % f)
                sys.stdout.flush()
            fdir,fbase = os.path.split(f)
            os.system(cmd + f + " " + os.path.join(tdir, fbase))
            copy(os.path.join(tdir, fbase), f)
            if options['verbose']: sys.stdout.write("done\n")
    finally:
        rmtree(tdir)


def arfx():
    """
arfx is used to move data in and out of ARF containers.

Usage: arfx [OPERATION] [OPTIONS] [FILES/ENTRIES]

Operations:
 -A: add data from one container to another
 -c: create a new container
 -r: append data to the container
 -t: list contents of the container
 -U: update metadata of entries
 -x: extract entries from the container
 -d: delete entries from the container

Options:
 -f FILE: use ARF file FILE
 -D URI:  specify the URI of the database for storing record IDs.
          None to use no database
 -v:      verbose output
 -n NAME: name entries sequentially, using NAME as the base
 -a ANIMAL: specify the animal
 -e EXPERIMENTER: specify the experimenter
 -p PROTOCOL: specify the protocol
 -s HZ:   specify the sampling rate of the data, in Hz
 -T DATATYPE: specify the data type (see --help-datatypes)
 -k KEY=VALUE: specifiy additional metadata
 -P:      when deleting entries, do not repack
 -u:      do not compress the data in the arf file
    """
    opts, args = getopt.gnu_getopt(sys.argv[1:], "AcrtUxdhf:D:vn:a:e:p:s:T:k:uP",
                                   ["help", "version","help-datatypes"])
    operation = None

    arffile = None

    for o,a in opts:
        # quasi operations
        if o in ('-h', '--help'):
            print arfx.__doc__
            sys.exit(0)
        elif o == '--version':
            print "%s version: %s" % (os.path.basename(sys.argv[0]), __version__)
            sys.exit(0)
        elif o == '--help-datatypes':
            print arf.DataTypes._doc()
            sys.exit(0)
        # main operations
        elif o == '-A':
            operation = 'copy'
        elif o == '-c':
            operation = 'create'
        elif o == '-r':
            operation = 'append'
        elif o == '-x':
            operation = 'extract'
        elif o == '-d':
            operation = 'remove'
        elif o == '-t':
            operation = 'list'
        elif o == '-U':
            operation = 'update'
        elif o == '-R':
            operation = 'reconcile'
        # options
        elif o == '-f':
            arffile = a
        elif o == '-D':
            if a == "None": a = None
            defaults['db_uri'] = a
        elif o == '-v':
            defaults['verbose'] = True
        elif o == '-n':
            defaults['entry_base'] = a
        elif o == '-a':
            defaults['entry_attrs']['animal'] = a
        elif o == '-e':
            defaults['entry_attrs']['experimenter'] = a
        elif o == '-p':
            defaults['entry_attrs']['protocol'] = a
        elif o == '-s':
            defaults['sampling'] = float(a)
        elif o == '-T':
            get_data_type(a)
        elif o == '-k':
            try:
                key,val = a.split('=')
                defaults['entry_attrs'][key] = val
            except ValueError:
                print >> sys.stderr, "-k %s argument badly formed; needs key=value" % a
        elif o == '-u':
            defaults['compress'] = None
        elif o == '-P':
            defaults['repack'] = False
        elif o == '--push':
            defaults['push_db'] = True
        elif o == '--pull':
            defaults['push_db'] = False

    if operation==None:
        print arfx.__doc__
        sys.exit(-1)

    if arffile==None:
        print "Error: must specify an ARF file (-f FILE)"
        sys.exit(-1)

    # parse arg list and dispatch - depends on operation
    elif operation in ('copy','create','append'):
        if len(args) < 1:
            print >> sys.stderr, "Error: must specify one or more input files."
            sys.exit(-1)
    if operation in ('list','extract','remove','append','reconcile','update'):
        if not os.path.exists(arffile):
            print >> sys.stderr, "Error: %s does not exist." % arffile
            sys.exit(-1)

    if operation == 'list':
        f = list_entries
    elif operation == 'extract':
        f = extract_entries
    elif operation == 'update':
        f = update_entries
    elif operation == 'remove':
        f = delete_entries
    elif operation == 'copy':
        f = copy_entries
    elif operation == 'append':
        f = add_entries
    elif operation == 'create':
        if os.path.exists(arffile):
            os.remove(arffile)
        f = add_entries

    try:
        f(arffile, args, **defaults)
    except Exception, e:
        raise
        print >> sys.stderr, e.message
        sys.exit(-1)

if __name__=="__main__":
    arfx()

# Variables:
# End:
