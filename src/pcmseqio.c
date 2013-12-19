/*
 * pcmseqio.c
 *
 * A python extension module that provides IO for the pcm_seq2 file format.
 * Data are stored in numpy arrays.
 *
 * Copyright (C) Dan Meliza, 2006-2009 (dmeliza@uchicago.edu)
 * Free for use under Creative Commons Attribution-Noncommercial-Share
 * Alike 3.0 United States License
 * (http://creativecommons.org/licenses/by-nc-sa/3.0/us/)
 *
 */

#include <Python.h>

#include "numpy/arrayobject.h"
#include "pcmseq.h"

typedef struct {
        PyObject_HEAD
        PCMFILE *pfp;
} PcmfileObject;

/* destructor */
static void
pcmfile_dealloc(PcmfileObject* self)
{
        if (self->pfp)
                pcm_close(self->pfp);
        self->ob_type->tp_free((PyObject*)self);
}

/* ctor */
static int
pcmfile_init(PcmfileObject* self, PyObject* args, PyObject *kwds)
{
        char *filename;
        char *mode = "r";

        // there's got to be a better way to ignore extra arguments. mode will
        // not get set properly if set as a keyword arg, but we don't support
        // writing here anyway
	if (!PyArg_ParseTuple(args, "s|s", &filename, &mode))
                return -1;

        // write mode is disabled for arfx
        if (strcmp(mode,"r") != 0) {
                PyErr_Format(PyExc_ValueError, "Writing to pcm_seq2 files is not supported");
                return -1;
        }

        if (self->pfp)
                pcm_close(self->pfp);

        if ((self->pfp = pcm_open(filename, mode)) == NULL) {
                PyErr_Format(PyExc_IOError, "Unable to open file %s", filename);
                return -1;
        }

        return 0;
}

/* methods */

static PyObject*
pcmfile_iter(PcmfileObject* self)
{
	return PySeqIter_New(PyObject_CallFunction((PyObject*) &PyRange_Type,
						   "lll", 1, self->pfp->nentries+1, 1));
}

static PyObject*
pcmfile_nentries(PcmfileObject* self, void *closure)
{
        return Py_BuildValue("i", self->pfp->nentries);
}

static PyObject*
pcmfile_filename(PcmfileObject* self, void *closure)
{
        return Py_BuildValue("s", self->pfp->name);
}

static PyObject*
pcmfile_timestamp(PcmfileObject* self, void *closure)
{
	double tstamp;
        struct pcmstat s;
        pcm_stat(self->pfp, &s);
	tstamp = (double)s.timestamp + (double)s.microtimestamp / 1e6;
        return Py_BuildValue("f", tstamp);
}

static int
pcmfile_settimestamp(PcmfileObject* self, PyObject *value, void *closure)
{
        double timestamp, integ;
	int sec, rv;
	long microsec;
        if (value == NULL) {
                PyErr_SetString(PyExc_TypeError, "Cannot delete the timestamp attribute");
                return -1;
        }

        timestamp = (double)PyFloat_AsDouble(value);
        if (timestamp <= 0) {
                PyErr_SetString(PyExc_TypeError, "Timestamp must be a positive float");
                return -1;
        }
	timestamp = modf(timestamp, &integ);
	sec = (int)integ;
	microsec = (long)(timestamp * 1e6);
        rv = pcm_ctl(self->pfp, PCMIOSETTIME, (int*)&sec);
	rv += pcm_ctl(self->pfp, PCMIOSETTIMEFRACTION, (long*)&microsec);
	if (rv < 0) {
		PyErr_SetString(PyExc_IOError, "Error setting timestamp");
		return -1;
	}
	return 0;
}

static PyObject*
pcmfile_samplerate(PcmfileObject* self, void *closure)
{
        struct pcmstat s;
        pcm_stat(self->pfp, &s);
        return Py_BuildValue("i", s.samplerate);
}

static int
pcmfile_setsamplerate(PcmfileObject* self, PyObject *value, void *closure)
{
        int srate;
        if (value == NULL) {
                PyErr_SetString(PyExc_TypeError, "Cannot delete the samplerate attribute");
                return -1;
        }

        srate = (int)PyInt_AsLong(value);
        if (srate <= 0) {
                PyErr_SetString(PyExc_TypeError, "Sample rate must be a positive integer");
                return -1;
        }

        return pcm_ctl(self->pfp, PCMIOSETSR, (int*)&srate);
}

static PyObject*
pcmfile_nsamples(PcmfileObject* self, void *closure)
{
        struct pcmstat s;
        pcm_stat(self->pfp, &s);
        return Py_BuildValue("i", s.nsamples);
}

static PyObject*
pcmfile_entry(PcmfileObject* self, void *closure)
{
        return Py_BuildValue("i", self->pfp->entry);
}

static int
pcmfile_seek(PcmfileObject* self, PyObject* value, void *closure)
{
        int entry;
        if (value == NULL) {
                PyErr_SetString(PyExc_TypeError, "Cannot delete the entry attribute");
                return -1;
        }

        entry = (int)PyInt_AsLong(value);
        if (pcm_seek(self->pfp, entry) != 0) {
                PyErr_SetString(PyExc_ValueError, "Invalid entry");
                return -1;
        }

        return 0;
}

static PyObject*
pcmfile_read(PcmfileObject* self, PyObject* args)
{
        /* allocate data */
        int nsamples;
        npy_intp shape[1];
        short *buf_p;
        PyArrayObject *pcmdata;

        if (pcm_read(self->pfp, &buf_p, &nsamples) == -1) {
                PyErr_SetString(PyExc_IOError, "Unable to read from file.");
                return NULL;
        }
        shape[0] = nsamples;

        pcmdata  = (PyArrayObject*) PyArray_SimpleNew(1,shape,NPY_SHORT);
        memcpy(PyArray_DATA(pcmdata), (void*)buf_p, nsamples * sizeof(short));

        return PyArray_Return(pcmdata);
}

// not used
static PyObject*
pcmfile_write(PcmfileObject* self, PyObject* args)
{
        PyObject* o;
        PyArrayObject* data;
	int rv;
        if (!PyArg_ParseTuple(args, "O", &o))
                return NULL;

        data = (PyArrayObject*) PyArray_FromAny(o, PyArray_DescrFromType(NPY_SHORT),
                                                1, 1, NPY_CONTIGUOUS, NULL);
        if (data==NULL)
                return NULL;

        rv = pcm_write(self->pfp, (short *)PyArray_DATA(data), PyArray_DIM(data, 0));
        Py_XDECREF(data);
	if (rv != 0) {
		PyErr_SetString(PyExc_IOError, "Unable to write to file.");
		return NULL;
	}
        return Py_BuildValue("");
}

static PyGetSetDef pcmfile_getseters[]={
        {"nentries", (getter)pcmfile_nentries, 0, "The number of entries in the file (r)", 0},
        {"filename", (getter)pcmfile_filename, 0, "The name of the file (r)", 0},
        {"sampling_rate", (getter)pcmfile_samplerate, (setter)pcmfile_setsamplerate,
         "The sample rate of the current entry (rw)", 0},
        {"nframes", (getter)pcmfile_nsamples, 0, "The number of samples in the current entry (r)",0},
        {"timestamp", (getter)pcmfile_timestamp, (setter)pcmfile_settimestamp,
         "The timestamp of the current entry (rw)",0},
        {"entry", (getter)pcmfile_entry, (setter)pcmfile_seek, "The current entry (set to seek to new entry)",0},
        {NULL}
};

static PyMethodDef pcmfile_methods[]= {
        {"read", (PyCFunction)pcmfile_read, METH_VARARGS|METH_KEYWORDS,
         "Read data from the current entry (r)"},
        {"write", (PyCFunction)pcmfile_write, METH_VARARGS,
         "Write data to the current entry (w)"},
        {NULL}
};

//static const char *pcmfile_classdoc = ;

static PyTypeObject PcmfileType = {
        PyObject_HEAD_INIT(NULL)
        0,                             /*ob_size*/
        "pcmseqio.pseqfile",              /*tp_name*/
        sizeof(PcmfileObject), /*tp_basicsize*/
        0,                             /*tp_itemsize*/
        (destructor)pcmfile_dealloc,                         /*tp_dealloc*/
        0,                         /*tp_print*/
        0,                         /*tp_getattr*/
        0,                         /*tp_setattr*/
        0,                         /*tp_compare*/
        0,                         /*tp_repr*/
        0,                         /*tp_as_number*/
        0,                         /*tp_as_sequence*/
        0,                         /*tp_as_mapping*/
        0,                         /*tp_hash */
        0,                         /*tp_call*/
        0,                         /*tp_str*/
        0,                         /*tp_getattro*/
        0,                         /*tp_setattro*/
        0,                         /*tp_as_buffer*/
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,        /*tp_flags*/
        "Reads or writes pcm_seq2 files.\n\n"
	"pcm_seq2 files can contain multiple entries; each entry has\n"
	"a timestamp and sampling rate associated with it.\n"
	"Only one bit depth is supported, 16-bit little-endian integers.\n\n"
	"pseqfile(filename,mode='r')\n\n"
	"mode can be 'r' or 'w'. Combined modes are not supported;\n"
	"many of the functions and properties raise errors or return\n"
	"meaningless results when accessed in write-only mode (indicated by\n"
	"'rw' below). Also, sampling rate and timestamp of new entries should be\n"
	"set before writing data to entries.", /* tp_doc*/
        0,                             /* tp_traverse */
        0,                             /* tp_clear */
        0,                             /* tp_richcompare */
        0,                             /* tp_weaklistoffset */
        pcmfile_iter,                             /* tp_iter */
        0,                             /* tp_iternext */
        pcmfile_methods,             /* tp_methods */
        0,             /* tp_members */
        pcmfile_getseters,                         /* tp_getset */
        0,                         /* tp_base */
        0,                         /* tp_dict */
        0,                         /* tp_descr_get */
        0,                         /* tp_descr_set */
        0,                         /* tp_dictoffset */
        (initproc)pcmfile_init,      /* tp_init */
        0,                         /* tp_alloc */
        0,                 /* tp_new */
};

static PyMethodDef _pcmseqio_methods[] = {
        {NULL}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC  /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
initpcmseqio(void)
{
        import_array();
        PyObject* m;

        PcmfileType.tp_dict = Py_BuildValue("{s:O,s:i,s:(s)}",
					    "dtype",(PyObject *)PyArray_DescrFromType(NPY_SHORT),
					    "nchannels", 1,
					    "channels", "pcm");

        PcmfileType.tp_new = PyType_GenericNew;
        if (PyType_Ready(&PcmfileType) < 0)
                return;

        m = Py_InitModule3("pcmseqio", _pcmseqio_methods,
			   "Read and write pcmseq2 files");

        Py_INCREF(&PcmfileType);
        PyModule_AddObject(m, "pseqfile", (PyObject *)&PcmfileType);
}
