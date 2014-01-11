# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""tool to import data from saber

Copyright (C) 2013 Dan Meliza <dmeliza@gmail.com>
Created Tue Jul 23 15:27:53 2013

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import re
import numpy as nx
import logging

import arf
from . import arfx
from . import io
from .tools import memoized

# template for target file
target_file_template = "{0}_{1}_{2}"
# dtype for event data
event_dtype = nx.dtype({'names': ("start", "status", "name"),
                        'formats': ('u8', 'u1', 'S256')})

_reg_create = re.compile(
    r"'(?P<file>(?P<base>\S+)_\w+.pcm_seq2)' (?P<action>created|closed)")
_reg_triggeron = re.compile(
    r"_ON, (?P<chan>\S+):entry (?P<entry>\d+) \((?P<onset>\d+)\)")
_reg_stimulus = re.compile(
    r"stimulus: REL:(?P<rel>[\d\.]*) ABS:(?P<abs>\d*) NAME:'(?P<stim>\S*)'")

log = logging.getLogger('arfxplog')   # root logger


@memoized
def get_uuid(pen, site, channel):
    """Returns a random uuid for a recording site/channel"""
    from uuid import uuid4
    ret = str(uuid4())
    log.info("assigned %s/%s/%s uuid: %s", pen, site, channel, ret)
    return ret


@memoized
def get_dest_arf(filename, dry_run):
    """Returns handle for destination arf file"""
    if dry_run:
        fp = arf.open_file(filename + ".arf", mode="a",
                           driver="core", backing_store=False)
    else:
        fp = arf.open_file(filename + ".arf", mode="w-")
        arf.set_attributes(
            fp, file_creator='org.meliza.arfx/arfxplog ' + arfx.__version__)
        log.info("opened '%s.arf' for writing", filename)
    return fp


def parse_explog(explog, entry_attrs, datatype, split_sites=False,
                 compression=1, channels=None, dry_run=False):
    """Parses an explog file to figure out where all the data is stored, and when
    everything happened. Creates one or more arf files to hold the data, and
    stores data under the associated entry.

    datatype:   specify the default type of data being recorded
                types may be specified in the explog; these have
                precedence
    channels:   if not None, only store data from these entries

    Additional arguments are used to set attributes on the newly created
    entries.

    """
    # look up source pcmseq2 file by channel name
    files = {}
    # dict of stimuli indexed by samplecount
    stimuli = {}
    # dataset attributes
    dset_attrs = {}
    # set of all onset times
    entries = {}
    fileonset = nx.uint64(0)  # corresponds to C long long type
    lastonset = nx.uint64(0)
    pen = 0
    site = 0

    efp = open(explog, 'rU')
    for line_num, line in enumerate(efp):
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
                    dset_attrs[fields[0]] = props
                except (AttributeError, ValueError):
                    log.warn(
                        "L%d parse error: bad channel metadata: ignoring", line_num)

        # file creation
        elif lstart == "FFFF":
            try:
                fname, base, action = _reg_create.search(line).groups()
            except AttributeError:
                log.warn("L%d parse error: %s", line_num, line)
                continue
            if channels is not None and base not in channels:
                continue
            if action == 'created':
                ifname = os.path.join(os.path.dirname(explog), fname)
                try:
                    files[base] = io.open(ifname, mode='r')
                except Exception as e:
                    log.warn(
                        "error opening source file '%s'; ARF files will be incomplete",
                        ifname)
                    log.debug(e)
            else:
                # file was closed; remove from list
                files.pop(base, None)

        # new pen or new site
        elif lstart == "IIII":
            fields = line.split()
            if fields[-2] == 'pen':
                pen = fields[-1]
            elif fields[-2] == 'site':
                site = fields[-1]

        # trigger lines
        elif lstart == "TTTT":
            if line.find("TRIG_OFF") > 0 or line.find("SONG_OFF") > 0:
                continue
            try:
                chan, entry, onset = _reg_triggeron.search(line).groups()
            except AttributeError:
                log.warn("L%d parse error: %s", line_num, line)
                continue
            if channels is not None and chan not in channels:
                continue
            try:
                ifp = files[chan]
            except KeyError:
                # user should already have been warned about missing data from
                # this file
                continue
            try:
                ifp.entry = int(entry)
            except ValueError:
                log.warn("L%d runtime error: unable to access %s/%d", line_num,
                         int(entry), chan)
                continue
            lastonset = nx.uint64(onset) + fileonset
            entry_name = "e%ld" % lastonset
            ofname = target_file_template.format(
                base, pen, site) if split_sites else base
            try:
                ofp = get_dest_arf(ofname, dry_run)
            except IOError:
                log.error("target file '%s' already exists; aborting", ofname)
                return -1
            log.debug("%s/%s -> %s/%s/%s", ifp.filename,
                      entry, ofp.filename, entry_name, chan)
            data = ifp.read()
            sampling_rate = ifp.sampling_rate

            if 'sampling_rate' in ofp.attrs:
                if ofp.attrs['sampling_rate'] != sampling_rate:
                    log.error("%s/%s sampling rate (%d) doesn't match target file (%d).\n"
                              "You may be attempting to load data from the wrong files!",
                              ifp.filename, entry, sampling_rate, ofp.attrs['sampling_rate'])
                    return -1
            else:
                ofp.attrs['sampling_rate'] = sampling_rate

            if lastonset in entries:
                entry = ofp[entry_name]
            else:
                entry = arf.create_entry(
                    ofp, entry_name,
                    ifp.timestamp,
                    sample_count=lastonset,
                    sampling_rate=sampling_rate,
                    entry_creator='org.meliza.arfx/arfxplog ' +
                    arfx.__version__,
                    pen=pen, site=site, **entry_attrs)
                entries[lastonset] = entry

            if chan in dset_attrs and 'datatype' in dset_attrs[chan]:
                chan_datatype = dset_attrs[chan]['datatype']
            else:
                chan_datatype = datatype

            dset = arf.create_dataset(entry, name=chan, data=data,
                                      datatype=chan_datatype, sampling_rate=sampling_rate,
                                      compression=compression,
                                      source_file=ifp.filename,
                                      source_entry=ifp.entry)
            # store duration of longest dataset; could also get this from
            # TRIG_OFF line, but this is a bit simpler.
            if data.size > entry.attrs.get('trial_off', 0):
                entry.attrs['trial_off'] = data.size
            arf.set_uuid(dset, get_uuid(pen, site, chan))

        # stimulus lines
        elif lstart == "QQQQ":
            try:
                rel, onset, stimname = _reg_stimulus.search(line).groups()
                lastonset = nx.uint64(onset) + fileonset
                if stimname.startswith('File='):
                    stimname = stimname[5:]
                stimuli[lastonset] = stimname
            except AttributeError:
                log.warn("L%d parse error: %s", line_num, line)

    # done parsing file
    efp.close()

    match_stimuli(stimuli, entries, sampling_rate=sampling_rate)


def match_stimuli(stimuli, entries, sampling_rate, table_name='stimuli'):
    """
    Create labels in arf entries that indicate stimulus onset and
    offset.  As the explog (or equivalent logfile) is parsed, the
    onset times of the entries and stimuli are collected.  Based on
    these times, this function matches each item in the list of
    stimuli to an entry.

    stimuli: dictionary of onset, stimulus_name pairs
    entries: dictionary of onset, arf entry pairs
    sampling_rate: the sampling rate of the onset times
    table_name:  the name of the node to store the label data in
    """
    log.debug("Matching stimuli to entries:")
    entry_times = nx.sort(list(entries.keys()))
    # slow, but simple
    for onset in sorted(stimuli.keys()):
        stim = stimuli[onset]
        idx = entry_times.searchsorted(onset, side='right') - 1
        if idx < 0 or idx > entry_times.size:
            log.debug("%s (onset=%d) -> no match!", stim, onset)
            continue

        eonset = entry_times[idx]
        entry = entries[eonset]

        units = 'samples'
        t_onset = onset - eonset

        # check that stim isn't occuring after the end of the recording
        max_length = max(dset.size for dset in entry.values())
        if t_onset >= max_length:
            log.debug("%s (onset=%d) -> after end of last entry", stim, onset)
            continue
        log.debug("%s (onset=%d) -> %s @ %d samples",
                  stim, onset, entry.name, t_onset)

        # add to list of intervals. this is trickier in h5py
        if table_name not in entry:
            stimtable = arf.create_table(
                entry, table_name, event_dtype,
                sampling_rate=sampling_rate,
                units=units, datatype=arf.DataTypes.EVENT)
            arf.set_uuid(
                stimtable, get_uuid(entry.attrs['pen'], entry.attrs['site'], table_name))
        else:
            stimtable = entry[table_name]
        arf.append_data(stimtable, (t_onset, 0x00, stim))
        entry.attrs['protocol'] = stim


def arfxplog():
    """Moves data from a saber experiment into ARF format."""
    import datetime
    import argparse

    p = argparse.ArgumentParser(
        description="Move data from a saber experiment into ARF format",)
    p.add_argument('--version', action='version',
                   version='%(prog)s ' + arfx.__version__)
    p.add_argument(
        '--dry-run', help="parse the explog but don't save the data to disk",
        action="store_true")
    p.add_argument('--help-datatypes',
                   help='print available datatypes and exit',
                   action='version', version=arf.DataTypes._doc())
    p.add_argument('-v', '--verbose', help="verbose output",
                   action="store_true")
    p.add_argument('-T', help='specify data type (see --help-datatypes)',
                   default=arf.DataTypes.UNDEFINED, metavar='DATATYPE',
                   dest='datatype', action=arfx.ParseDataType)
    p.add_argument(
        '-k', help='specify attributes of entries', action=arfx.ParseKeyVal,
        metavar="KEY=VALUE", dest='attrs', default={})
    p.add_argument('-s', help="generate arf file for each pen/site",
                   action="store_true", dest='split')
    p.add_argument('-a', help="specify the animal in the experiment",
                   dest='animal')
    p.add_argument('-e', help="specify the experimenter", dest='experimenter')
    p.add_argument('-z', help="compression level (0-9)", type=int, default=1,
                   dest="compression")
    p.add_argument(
        '--chan', help='restrict to specific channels (comma-delimited list)',
        dest="channels")
    p.add_argument("explog", help="explog generated by saber (note: pcm_seq2 files must "
                   "be in the same directory)")

    opts = p.parse_args()

    ch = logging.StreamHandler()
    formatter = logging.Formatter("[%(name)s] %(message)s")
    if opts.verbose:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO
    log.setLevel(loglevel)
    ch.setLevel(loglevel)  # change
    ch.setFormatter(formatter)
    log.addHandler(ch)
    log.info("version: %s", arfx.__version__)
    log.info("run time: %s", datetime.datetime.now())

    if opts.split:
        log.info("data will be split into per-site arf files")
    if opts.channels is not None:
        opts.channels = opts.channels.split(",")
        log.info("data will only be stored from channels: %s", opts.channels)
    opts.attrs['animal'] = opts.animal
    opts.attrs['experimenter'] = opts.experimenter

    return parse_explog(
        opts.explog, opts.attrs, opts.datatype, split_sites=opts.split,
        compression=opts.compression, channels=opts.channels,
        dry_run=opts.dry_run)

# Variables:
# End:
