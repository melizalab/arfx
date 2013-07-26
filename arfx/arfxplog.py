# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""tool to import data from saber

Copyright (C) 2013 Dan Meliza <dmeliza@gmail.com>
Created Tue Jul 23 15:27:53 2013
"""

import os
import sys
import numpy as nx

import arf
from . import io
from arfx import __version__

defaults = {
    'verbose': False,
    'entry_base': None,
    'sampling': 20000,
    'datatype': arf.DataTypes.UNDEFINED,
    'compress': 1,
    'repack': True,
    'split_sites': False,
    'entry_attrs': {},
}
# template for extracted files
default_extract_template = "{entry}_{channel}"
# template for created entries
default_entry_template = "{base}_{index:04}"


def get_data_type(a):
    if a.isdigit():
        defaults['datatype'] = int(a)
    else:
        defaults['datatype'] = getattr(arf.DataTypes, a.upper(), None)
        if defaults['datatype'] is None:
            print >> sys.stderr, "Error: %s is not a valid data type" % a
            print >> sys.stderr, arf.DataTypes._doc()
            sys.exit(-1)


class arfgroup(object):

    """
    Cache handles to one or more arf files. Arf files are indexed by
    pen,site.  If split_sites is true, then each site refers to a
    separate file.
    """
    _template = "%s_%d_%d"

    def __init__(self, basename, **options):
        """
        basename:  the basename of the arf files in the group

        Additional options
        ------------------
        db_uri:        specify alternate location of the entry database
        split_sites:   split data for each site into multiple files (default False)
        open_mode:     mode string to open new files with (default 'w')
        """
        self.basename = basename
        self.open_mode = options.get('open_mode', 'w')
        self.factory = lambda x: arf.file(x, mode=self.open_mode)

        if options.get('split_sites', False):
            self.handles = {}
        else:
            self.handles = self._openfile(basename)

    def _openfile(self, basename):
        print "Creating %s.arf" % basename
        fp = self.factory(basename + ".arf")
        fp.set_attributes(program='arfxplog')
        return fp

    def __getitem__(self, key):
        if not isinstance(self.handles, dict):
            return self.handles
        basename = self._template % (self.basename, key[0], key[1])
        if basename not in self.handles:
            self.handles[basename] = self._openfile(basename)
        return self.handles[basename]

    def __iter__(self):
        if isinstance(self.handles, dict):
            return self.handles.itervalues()
        else:
            return (h for h in (self.handles,))


def parse_explog(logfile, **options):
    """
    Parse an explog file to figure out where all the data is stored,
    and when everything happened. Creates one or more arf files to
    hold the data, and stores data under the associated entry.

    verbose:    print information about what's happening
    datatype:   specify the default type of data being recorded
                types may be specified in the explog; these have
                precedence
    channels:   if not None, only store data from these entries
    skip_data:  if True, don't actually store the data (for debugging)

    Additional arguments are used to set attributes on the newly
    created entries.
    """
    import re
    channels = options.get('channels', None)
    skip_data = options.get('skip_data', False)

    _reg_create = re.compile(
        r"'(?P<file>(?P<base>\S+)_\w+.pcm_seq2)' (?P<action>created|closed)")
    _reg_triggeron = re.compile(
        r"_ON, (?P<chan>\S+):entry (?P<entry>\d+) \((?P<onset>\d+)\)")
    _reg_stimulus = re.compile(
        r"stimulus: REL:(?P<rel>[\d\.]*) ABS:(?P<abs>\d*) NAME:'(?P<stim>\S*)'")

    # look up file by channel
    filenames = {}
    files = {}
    # dict of stimuli indexed by samplecount
    stimuli = {}
    # dict of channel properties from the file
    chanprops = {}
    # set of all onset times
    entries = {}
    fileonset = nx.uint64(0)  # corresponds to C long long type
    lastonset = nx.uint64(0)
    currentpen = 0
    currentsite = 0

    user_attributes = options['entry_attrs']

    fp = open(logfile, 'rt')
    arfhandler = arfgroup(os.path.splitext(logfile)[0], **options)
    for line_num, line in enumerate(fp):
        lstart = line[0:4]

        # control info
        if lstart == '%%%%':
            if line.rstrip().endswith('start'):
                fileonset = lastonset
            elif line.find('add') > -1:
                try:
                    fields = line.partition('add')[-1].split()
                    props = dict(f.split('=') for f in fields[1:])
                    if 'datatype' in props:
                        props['datatype'] = getattr(
                            arf.DataTypes, props['datatype'].upper())
                    chanprops[fields[0]] = props
                except (AttributeError, ValueError):
                    print >> sys.stderr, "parse error: bad channel metadata (line %d): ignoring " % line_num

        # file creation
        elif lstart == "FFFF":
            try:
                fname, base, action = _reg_create.search(line).groups()
                if channels is not None and base not in channels:
                    continue
                if action == 'created':
                    try:
                        files[base] = io.open(fname, mode='r')
                        filenames[base] = fname
                    except Exception, e:
                        print >> sys.stderr, "error opening file %s; ARF files will be incomplete" % fname
                else:
                    filenames.pop(base, None)
                    files.pop(base, None)
            except AttributeError, e:
                print >> sys.stderr, "parse error: Unparseable FFFF line (%d): %s" % (
                    line_num, line)

        # new pen or new site
        elif lstart == "IIII":
            if line.find('pen') > 0:
                currentpen = int(line.split()[-1])
            elif line.find('site') > 0:
                currentsite = int(line.split()[-1])

        # trigger lines
        elif lstart == "TTTT":
            if line.find("TRIG_OFF") > 0 or line.find("SONG_OFF") > 0:
                continue
            try:
                chan, entry, onset = _reg_triggeron.search(line).groups()
                if channels is not None and chan not in channels:
                    continue
                ifp = files[chan]
                ifp.entry = int(entry)
                lastonset = nx.uint64(onset) + fileonset
                entry_name = "e%ld" % lastonset
                ah = arfhandler[currentpen, currentsite]
                if options.get('verbose', False):
                    sys.stdout.write("%s/%s -> %s/%s" %
                                     (filenames[chan], entry, ah.h5.filename, entry_name))
                if lastonset in entries:
                    entry = ah[entry_name]
                else:
                    entry = ah.create_entry(
                        entry_name, ifp.timestamp, sample_count=lastonset,
                        pen=currentpen, site=currentsite, **user_attributes)
                    entries[lastonset] = entry

                if skip_data:
                    data = nx.zeros(1)
                else:
                    data = ifp.read()
                sampling_rate = ifp.sampling_rate

                if chan in chanprops and 'datatype' in chanprops[chan]:
                    datatype = chanprops[chan]['datatype']
                else:
                    datatype = options['datatype']

                entry.add_data(name=chan, data=data,
                               datatype=datatype, sampling_rate=sampling_rate,
                               compression=options[
                                   'compress'], source_file=ifp.filename,
                               source_entry=ifp.entry)

                entry.attrs[
                    'max_length'] = max(entry.attrs.get('max_length', 0.0),
                                        1.0 * data.size / sampling_rate)
                if options.get('verbose', False):
                    sys.stdout.write("/%s\n" % chan)
            except AttributeError, e:
                print >> sys.stderr, "line %d: Unparseable TTTT line: %s" % (
                    line_num, line)
            except KeyError, e:
                print >> sys.stderr, "line %d: TRIG_ON references channel %s, but I can't find the file for it\n(%s)" % \
                    (line_num, chan, e)
            except ValueError, e:
                print >> sys.stderr, "line %d: Error accesing entry %d in %s" % (
                    line_num, int(entry), chan)

        # stimulus lines
        elif lstart == "QQQQ":
            try:
                rel, onset, stimname = _reg_stimulus.search(line).groups()
                lastonset = long(onset) + fileonset
                if stimname.startswith('File='):
                    stimname = stimname[5:]
                stimuli[lastonset] = stimname
            except AttributeError, e:
                print >> sys.stderr, "parse error: Unparseable QQQQ line (%d): %s" % (
                    line_num, line)

    # done parsing file
    fp.close()

    if options.get('verbose', False):
        sys.stdout.write("Matching stimuli to entries:\n")
    match_stimuli(stimuli, entries, verbose=options.get('verbose', False))
    check_samplerates(arfhandler)


def match_stimuli(stimuli, entries, table_name='stimuli', verbose=False):
    """
    Create labels in arf entries that indicate stimulus onset and
    offset.  As the explog (or equivalent logfile) is parsed, the
    onset times of the entries and stimuli are collected.  Based on
    these times, this function matches each item in the list of
    stimuli to an entry.

    stimuli: dictionary of onset,stimulus_name pairs
    entries: dictionary of onset, arf entry pairs
    table_name:  the name of the node to store the label data in
    verbose:     if true, print debug info about matches
    """
    entry_times = nx.sort(entries.keys())
    # slow, but simple
    for onset in sorted(stimuli.keys()):
        stim = stimuli[onset]
        if verbose:
            sys.stdout.write("%s (onset=%d) -> " % (stim, onset))
        idx = entry_times.searchsorted(onset, side='right') - 1
        if idx < 0 or idx > entry_times.size:
            if verbose:
                sys.stdout.write("no match!\n")
            continue

        eonset = entry_times[idx]
        entry = entries[eonset]

        sampling_rate = 1.0
        units = 'samples'
        for node in entry.itervalues():
            # this assumes a single sampling rate for all data
            if 'sampling_rate' in node.attrs:
                sampling_rate = node.attrs['sampling_rate']
                units = 's'
                break
        t_onset = float(onset - eonset) / sampling_rate
        if verbose:
            sys.stdout.write("%s @ %.4f %s" % (entry.name, t_onset, units))

        # check that stim isn't occuring after the end of the recording
        if t_onset >= entry.attrs.get('max_length', nx.inf):
            if verbose:
                sys.stdout.write(" :: after end of entry, skipping\n")
            continue

        # add to list of intervals. this is trickier in h5py
        if table_name not in entry:
            stimtable = arf.table.create_table(
                entry, table_name, arf._interval_dtype,
                units=units, datatype=arf.DataTypes.STIMI)
        else:
            stimtable = arf.table(entry[table_name])
        t_onset = float(onset - eonset) / sampling_rate
        stimtable.append((stim, t_onset, t_onset))
        entry.attrs['protocol'] = stim
        if verbose:
            sys.stdout.write("\n")


def check_samplerates(arfps):
    """
    For each arf in arfps, iterate through all sampled data nodes and
    collate sampling rate information.  For saber-generated data,
    these should all be equal.  Check for this, and set a global
    sampling-rate attribute on the file.
    """
    for arfp in arfps:
        srates = []
        for entry in arfp:
            for dset in arfp[entry].itervalues():
                if 'sampling_rate' in dset.attrs:
                    srates.append(dset.attrs['sampling_rate'])
        for sr in srates:
            if sr != srates[0]:
                print >> sys.stderr, "warning: %s has nodes with varying sampling rates" % (
                    arfp.h5.filename)
                break
        arfp.set_attributes(sampling_rate=srates[0])


def arfxplog():
    """
    Move data from a saber experiment into ARF format.

    Usage: arfxplog [OPTIONS] <file.explog>

    Options:
     -v:              verbose output
     -a ANIMAL:       specify the animal
     -e EXPERIMENTER: specify the experimenter
     -T DATATYPE:     specify the data type (integer or code)
     -k KEY=VALUE:    specifiy additional metadata
     --chan CHANS:    restrict to specific channels (comma-delimited)
     -s:              generate arf file for each pen/site

     --help-datatypes: display documentation on available data types
     --version: display version information
    """
    import getopt

    channels = None
    opts, args = getopt.getopt(sys.argv[1:], 'hva:e:T:k:s',
                               ['chan=', 'help', 'version', 'help-datatypes', "debug"])
    for o, a in opts:
        if o in ('-h', '--help'):
            print arfxplog.__doc__
            sys.exit(0)
        elif o == '--version':
            print "%s version: %s" % (os.path.basename(sys.argv[0]), __version__)
            sys.exit(0)
        elif o == '--help-datatypes':
            print arf.DataTypes._doc()
            sys.exit(0)
        elif o == '-v':
            defaults['verbose'] = True
        elif o == '-a':
            defaults['entry_attrs']['animal'] = a
        elif o == '-e':
            defaults['entry_attrs']['experimenter'] = a
        elif o == '-T':
            get_data_type(a)
        elif o == '-k':
            try:
                key, val = a.split('=')
                defaults['entry_attrs'][key] = val
            except ValueError:
                print >> sys.stderr, "-k %s argument badly formed; needs key=value" % a
        elif o == '--chan':
            channels = a.split(',')
        elif o == '-s':
            defaults['split_sites'] = True
        elif o == '--debug':
            defaults['skip_data'] = True

    if len(args) < 1:
        print arfxplog.__doc__
        sys.exit(-1)

    parse_explog(args[0], channels=channels, **defaults)


# Variables:
# End:
