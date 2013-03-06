# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
Convert old ARF formats to the current one.

Copyright (C) 2011 Daniel Meliza <dmeliza@dylan.uchicago.edu>
Created 2011-04-06
"""

import sys
import h5py as hp
import arf
from . import h5vlen
from distutils.version import StrictVersion

def unpickle_attrs(attrs, old_attrs=None):
    """
    Some attributes in pytables get pickled (including numeric
    types). Try to unpickle them and store them in an h5py compatible
    format.  If old_attrs is set, copies any missing attributes in
    this mapping to attrs, unpickling as needed.
    """
    from itertools import chain
    from cPickle import loads

    if old_attrs is not None:
        olda = dict((k,v) for k,v in old_attrs.iteritems() if k not in attrs)
    else:
        olda = {}
    for k,v in chain(attrs.iteritems(), olda.iteritems()):
        if isinstance(v,basestring) and len(v) > 0 and v[-1]=='.':
            try:
                val = loads(v)
            except:
                print "error unpickling attribute %s (%s)" % (k,v)
                continue

            try:
                attrs[k] = val
            except:
                print "error storing pickled attribute %s" % (k,v)
                attrs[k] = v


def cvt_11(afp, **kwargs):
    """
    Convert a 0.9 or 1.0 file to 1.1.  The catalogs no longer exist,
    so we iterate through the records and use it to update the groups
    and datasets.  Furthermore, single-channel datasets are the norm,
    so if there are any multi-column datasets they need to be split.

    Format differences between 0.9 and 1.0:
    - _Catalog schema has different type for timestamp (now seconds since Epoch)
       and additional fields (group, timestamp_m)
    - _Channel schema has different names (node instead of table) and an
      additional column (datatype)
    """
    _catalogname = "catalog"

    print "* Upgrading %s to 1.1" % afp.filename
    recuri = afp.attrs.get('database_uri',None)

    ecat = afp[_catalogname][::]
    print "** Upgrading entries"
    for r in ecat:
        entry = afp[r['name']]
        for k in ('recid','animal','experimenter','protocol'):
            if k in ecat.dtype.names: entry.attrs[k] = r[k]
        # timestamp requires special processing
        ts,tsm = r['timestamp'], 0
        if 'timestamp_m' in r.dtype.names: tsm = r['timestamp_m']
        entry.attrs['timestamp'] = arf.convert_timestamp((ts,tsm))

        if recuri is not None:
            entry.attrs['recuri'] = recuri

        if not _catalogname in entry:
            print "** %s doesn't have a catalog: already upgraded?" % entry.name
            continue
        if not len(entry) > 1:
            print "** %s appears to be empty" % entry.name
            del entry[_catalogname]
            continue
        ccat = entry[_catalogname][::]
        print "** Upgrading datasets for %s:" % entry.name
        # first rename all old nodes to avoid collisions
        old_nodes = set()
        for dset_name in entry.keys():
            new_name = "_arfmigrate_" + dset_name
            entry[new_name] = entry[dset_name]
            #print "renaming %s to %s" % (dset_name, new_name)
            del entry[dset_name]
            old_nodes.add(new_name)
        # then iterate through channels
        for rr in ccat:
            node_name = ('node' in ccat.dtype.names) and rr['node'] or rr['table']
            datatype = ('datatype' in ccat.dtype.names) and rr['datatype'] or arf.DataTypes.UNDEFINED
            old_dset = entry["_arfmigrate_" + node_name]
            dtype,stype,ncol = arf.dataset_properties(old_dset)
            if dtype=='event' and stype=='vlarray':
                # vlarrays are converted to regular arrays
                print "    %s/%d (vlarray) -> %s" % (node_name, rr['column'], rr['name'])
                data = h5vlen.read(old_dset)[rr['column']]
                new_dset = entry.create_dataset(rr['name'], data=data,
                                                maxshape=None, chunks=True, compression=False)
            elif dtype=='sampled' and ncol > 1:
                # multi-column datasets are split
                print "    %s/%d (ndarray) -> %s" % (node_name, rr['column'], rr['name'] )
                data = old_dset[:,rr['column']]
                sampling_rate = old_dset.attrs['sampling_rate']
                # don't compress until file is repacked
                new_dset = entry.create_dataset(rr['name'], data=data,
                                                maxshape=None, chunks=True, compression=False)
                arf.set_attributes(new_dset, sampling_rate=sampling_rate)

            else:
                # all other datatypes can simply be renamed
                print "    %s (%s) -> %s" % (node_name, dtype, rr['name'])
                new_dset = entry[rr['name']] = old_dset

            arf.set_attributes(new_dset, datatype=datatype, units=rr['units'])
            unpickle_attrs(new_dset.attrs,old_dset.attrs)

        print "    Removing old datasets"
        for n in old_nodes:
            del entry[n]
        unpickle_attrs(entry.attrs)

    print "** Removing top-level catalog"
    del afp[_catalogname]
    print "** Fixing top-level attributes"
    afp.attrs['arf_version'] = '1.1'
    if 'database_class' in afp.attrs: del afp.attrs['database_class']
    if 'database_uri' in afp.attrs: del afp.attrs['database_uri']
    unpickle_attrs(afp.attrs)

# these are removed in version 2.0
_pytables_attributes = \
    dict(file=dict(TITLE='',VERSION='1.0',CLASS='GROUP',PYTABLES_FORMAT_VERSION='2.0'),
         group=dict(TITLE='',VERSION='1.0',CLASS='GROUP'),
         carray=dict(TITLE='',VERSION='1.0',CLASS='CARRAY'),
         earray=dict(TITLE='',VERSION='1.3',CLASS='EARRAY',EXTDIM=1),
         vlarray=dict(TITLE='',VERSION='1.3',CLASS='VLARRAY'),
         table=dict(TITLE='',VERSION='2.6',CLASS='TABLE')) # also need field names


def cvt_11_20(afp, **kwargs):
    """ convert a version 1.1 ARF file to version 2.0 """
    from uuid import uuid4

    pyt_attr_names = set(k for v in _pytables_attributes.values() for k in v.keys())

    sampling_rate = kwargs.get("sampling_rate",None)

    print "* Upgrading %s to 2.0" % afp.filename

    for ename,entry in afp.items():
        print "** Upgrading attributes for", ename
        if "uuid" not in entry.attrs:
            print "*** Adding uuid"
            arf.set_uuid_attr(entry)
        for k in entry.attrs:
            if k in pyt_attr_names: del entry.attrs[k]

        for dname,dset in entry.items():
            for k in dset.attrs:
                if k in pyt_attr_names: del dset.attrs[k]
            if "units" in dset.attrs and dset.attrs["units"]=="samples" and "sampling_rate" not in dset.attrs:
                if sampling_rate is None:
                    raise ValueError, "need sampling rate for %s; supply as metadata" % dset.name
                print "*** Adding sampling rate to %s" % dname
                dset.attrs['sampling_rate'] = float(sampling_rate)

    print "** Fixing top-level attributes"
    for k in afp.attrs:
        if k in pyt_attr_names: del afp.attrs[k]
    afp.attrs['arf_version'] = '2.0'

converters = [(StrictVersion('1.1'), cvt_11),
              (StrictVersion('2.0'), cvt_11_20)]

def migrate_file(path, newname=None, **kwargs):
    """ upgrade <path> to current version of ARF. If <newname> is not None, does this on a copy """
    from shutil import copy2
    if newname is not None:
        copy2(path, newname)
        path = newname

    fp = hp.File(path,'r+')
    for v,f in converters:
        file_version = StrictVersion(fp.attrs.get('arf_version',"0.9"))
        if file_version < v:
            f(fp, **kwargs)
    print "* %s is up to date" % path
    fp.close()

# Variables:
# End:
