# some cdefs, lifted from h5py
cdef extern from "stdlib.h":
  ctypedef long size_t
  void *malloc(size_t size)
  void free(void *ptr)

cdef extern from "hdf5.h":

  # Basic types
  ctypedef int hid_t
  ctypedef int hbool_t
  ctypedef int herr_t
  ctypedef int htri_t
  ctypedef long long hsize_t
  ctypedef signed long long hssize_t
  ctypedef signed long long haddr_t

  ctypedef struct hvl_t:
    size_t len                 # Length of VL data (in base type units)
    void *p                    # Pointer to VL data

  int H5P_DEFAULT
  int H5S_ALL, H5S_MAX_RANK

  hid_t H5Dget_space(hid_t dset_id)
  herr_t H5Dread(hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id, hid_t plist_id, void *buf) except *
  herr_t H5Dwrite(hid_t dset_id, hid_t mem_type, hid_t mem_space, hid_t file_space, hid_t xfer_plist, void* buf) except *
  herr_t H5Dvlen_reclaim(hid_t type_id, hid_t space_id,  hid_t plist, void *buf)
  hid_t H5Screate_simple(int rank, hsize_t *dims, hsize_t *maxdims)
  herr_t H5Sclose(hid_t space_id)
