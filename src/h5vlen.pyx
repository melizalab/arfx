# -*- coding: iso-8859-1 -*-
# -*- mode: python -*-
"""
Code to read and write VLEN objects under h5py.  Support is mostly for
compatibility, since most entries should have only a handful of
event-type channels with clearly distinct identities.  The vlen
datatype is only really necessary when storing multi-repeat data in a
single entry (e.g. all of a neuron's responses to various stimuli),
which is somewhat outside the specification of the ARf format.

Data must be read all at once, or written all at once.
"""


# write:
# create dataset with appropriate type
# allocate write structure. iterate and assign pointers to buffers
# call H5Dwrite
# deallocate write structure

import numpy as nx
cimport numpy as nx
nx.import_array()

from h5defs cimport *
from h5py._objects import ObjectID
from h5py.h5t import TypeVlenID, TypeAtomicID, py_create, vlen_create
from h5py import h5d, h5s, h5p
from h5py._hl import dataset
from cpython cimport PyObject

def read(dset):
    """
    Read a dataset with variable-length data type.  The vlen datatype
    must contain an atomic type (no nesting) and the dataspace must be
    one-dimensional.  All the elements of the dataset are returned as
    a list, with each element represented by a numpy array.
    """
    cdef int i, narr
    cdef hid_t space_id
    cdef hvl_t *rdata = NULL
    cdef nx.npy_intp dims[1]
    cdef nx.ndarray arr

    # check for the correct data type
    if isinstance(dset, dataset.Dataset):
        dset = dset.id
    vtype = dset.get_type()
    if not isinstance(vtype, TypeVlenID):
        raise ValueError, "Dataset type is not vlen"
    dtype = vtype.get_super()
    if not isinstance(dtype, TypeAtomicID):
        raise ValueError, "Dataset can only contain vlens of atomic type"
    if len(dset.shape)>1:
        raise ValueError, "Can only read from 1D vlen arrays"
    cdef int typenum = dtype.dtype.num

    # allocate read buffer, which is just an array of lengths and pointers
    narr = dset.shape[0]
    rdata = <hvl_t*>malloc(sizeof(hvl_t)*narr)

    # read the data
    space_id = H5Dget_space(dset.id);
    H5Dread(dset.id, vtype.id, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdata)

    # for each element, wrap buffer in a numpy array, and copy
    out = []
    for i in range(narr):
        dims[0] = rdata[i].len
        arr = nx.PyArray_SimpleNewFromData(1, dims, typenum, rdata[i].p)
        out.append(arr.copy())

    # deallocate read buffer
    H5Dvlen_reclaim(vtype.id, space_id, H5P_DEFAULT, rdata)
    free(rdata)
    H5Sclose(space_id)

    return out

def create(loc, object name not None, object dtype not None):
    """ (ObjectID loc, STRING name, DTYPE) => DatasetID
    Create a dataset to store vlen elements of type <dtype>. The
    dataspace is 1D, and extensible, so additional data can be added
    with write().
    """
    space = h5s.create_simple((0,),(h5s.UNLIMITED,))
    vtype = vlen_create(py_create(dtype))
    # have to set chunking in order to be extensible
    plist = h5p.create(h5p.DATASET_CREATE)
    plist.set_chunk((1,))
    plist.set_fill_time(h5d.FILL_TIME_ALLOC)
    dsid = h5d.create(loc, name, vtype, space, plist)
    return dataset.Dataset(dsid)


def write(dset, data not None):
    """
    Write a sequence of ndarrays as a vlen dataset. This is not an
    encouraged format for event data, which ought to be stored as 1D
    arrays, one for each channel.
    """
    cdef int i, narr, nold
    cdef hvl_t wdata
    cdef nx.npy_intp dims[1]

    if not isinstance(data, (tuple,list)) or not all(isinstance(x,nx.ndarray) for x in data):
        raise TypeError, "Data must be a sequence of ndarrays"
    narr = len(data)

    dims[0] = 1
    memspace = h5s.create_simple((1,))

    # expand dataset then add data one by one
    dsid = dset.id
    nold = dsid.shape[0]
    dsid.extend((nold+narr,))
    fspace = dsid.get_space()

    for i,x in enumerate(data):
        vtype = vlen_create(py_create(x.dtype))
        wdata.len = x.size
        wdata.p = nx.PyArray_DATA(x)
        fspace.select_elements([(nold+i,)])
        H5Dwrite(dsid.id,vtype.id,memspace.id,fspace.id,H5P_DEFAULT,&wdata)

