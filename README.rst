arfx
----

|ProjectStatus|_ |Version|_ |BuildStatus|_ |License|_ |PythonVersions|_

.. |ProjectStatus| image:: https://www.repostatus.org/badges/latest/active.svg
.. _ProjectStatus: https://www.repostatus.org/#active

.. |Version| image:: https://img.shields.io/pypi/v/arfx.svg
.. _Version: https://pypi.python.org/pypi/arfx/

.. |BuildStatus| image:: https://github.com/melizalab/arfx/actions/workflows/python-package.yml/badge.svg
.. _BuildStatus: https://github.com/melizalab/arfx/actions/workflows/python-package.yml

.. |License| image:: https://img.shields.io/pypi/l/arfx.svg
.. _License: https://opensource.org/license/bsd-3-clause/

.. |PythonVersions| image:: https://img.shields.io/pypi/pyversions/arfx.svg
.. _PythonVersions: https://pypi.python.org/pypi/arfx/

**arfx** is a family of commandline tools for copying sampled data in
and out of ARF containers. ARF (https://github.com/melizalab/arf) is an
open, portable file format for storing behavioral and neural data, based
on `HDF5 <http://www.hdfgroup.org/HDF5>`__.

installation
~~~~~~~~~~~~

.. code:: bash

   pip install arfx

or from source:

.. code:: bash

   python setup.py install

usage
~~~~~

The general syntax is ``arfx operation [options] files``. The syntax is
similar to ``tar``. Operations are as follows:

-  **-A:** copy data from one container to another
-  **-c:** create a new container
-  **-r:** append data to the container
-  **-t:** list contents of the container
-  **-x:** extract entries from the container
-  **-d:** delete entries from the container

Options specify the target ARF file, verbosity, automatic naming
schemes, and any metadata to be stored in the entry.

-  **-f FILE:** use ARF file FILE
-  **-v:** verbose output
-  **-n NAME:** name entries sequentially, using NAME as the base
-  **-a ANIMAL:** specify the animal
-  **-e EXPERIMENTER:** specify the experimenter
-  **-p PROTOCOL:** specify the protocol
-  **-s HZ:** specify the sampling rate of the data, in Hz
-  **-T DATATYPE:** specify the type of data
-  **-u:** do not compress data in the arf file
-  **-P:** when deleting entries, do not repack

input files
           

**arfx** can read sampled data from ``pcm``, ``wave``, ``npy`` and
``mda`` files. Support for additional file formats can be added as
plugins (see 4).

When adding data to an ARF container (``-c`` and ``-r`` modes), the
input files are specified on the command line, and added in the order
given. By default, entries are given the same name as the input file,
minus the extension; however, if the input file has more than one entry,
they are given an additional numerical extension. To override this, the
``-n`` flag can be used to specify the base name; all entries are given
sequential names based on this.

The ``-n, -a, -e, -p, -s, -T`` options are used to store information
about the data being added to the file. The DATATYPE argument can be the
numerical code or enumeration code (run ``arfx --help-datatypes`` for a
list), and indicates the type of data in the entries. All of the entries
created in a single run of arfx are given these values. The ``-u``
option tells arfx not to compress the data, which can speed up I/O
operations slightly.

Currently only one sampled dataset per entry is supported. Clearly this
does not encompass many use cases, but **arfx** is intended as a simple
tool. More specialized import procedures can be easily written in Python
using the ``arf`` library.

output files
            

The entries to be extracted (in ``-x`` mode) can be specified by name.
If no names are specified, all the entries are extracted. All sampled
datasets in each entry are extracted as separate channels, because they
may have different sampling rates. Event datasets are not extracted.

By default the output files will be in ``wave`` format and will have
names with the format ``entry_channel.wav``. The ``-n`` argument can be
used to customize the names and file format of the output files. The
argument must be a template in the format defined by the `python string
module <http://docs.python.org/library/string.html###format-specification-mini-language>`__.
Supported field names include ``entry``, ``channel``, and ``index``, as
well as the names of any HDF5 attributes stored on the entry or channel.
The extension of the output template is used to determine the file
format. Currently only ``wave`` is supported, but additional formats may
be supplied as plugins (see 4).

The metadata options are ignored when extracting files; any metadata
present in the ARF container that is also supported by the target
container is copied.

other operations
                

As with ``tar``, the ``-t`` operation will list the contents of the
archive. Each entry/channel is listed on a separate line in path
notation.

The ``-A`` flag is used to copy the contents of one ARF file to another.
The entries are copied without modification from the source ARF file(s)
to the target container.

The ``-d`` (delete) operation uses the same syntax as the extract
operation, but instead of extracting the entries, they are deleted.
Because of limitations in the underlying HDF5 library, this does not
free up the space, so the file is repacked unless the ``-P`` option is
set.

The ``-U`` (update) operation can be used to add or update attributes of
entries, and to rename entries (if the ``-n`` flag is set).

The ``--write-attr`` operation can be used to store the contents of text
files in top-level attributes. The attributes have the name
``user_<filename>``. The ``--read-attr`` operation can be used to read
out those attributes. This is useful when data collection programs
generate log or settings files that you want to store in the ARF file.

other utilities
~~~~~~~~~~~~~~~

This package comes with a few additional scripts that do fairly specific
operations.

arfx-split
^^^^^^^^^^

This script is used to reorganize very large recordings, possibly
contained in multiple files, into manageable chunks. Each new entry is
given an updated timestamp and attributes from the source entries.
Currently, no effort is made to splice data across entries or files.
This may result in some short entries. Only sampled datasets are
processed.

arfx-collect-sampled
^^^^^^^^^^^^^^^^^^^^

This script is used to export data into a flat binary structure. It
collects sampled data across channels and entries into a single 2-D
array. The output can be stored in a multichannel wav file or in a raw
binary ``dat`` format (N samples by M channels), which is used by a wide
variety of spike-sorting tools.

extending arfx
~~~~~~~~~~~~~~

Additional formats for reading and writing can be added using the Python
setuptools plugin system. Plugins must be registered in the ``arfx.io``
entry point group, with a name corresponding to the extension of the
file format handled by the plugin.

An arfx IO plugin is a class with the following required methods:

``__init__(path, mode, **attributes)``: Opens the file at ``path``. The
``mode`` argument specifies whether the file is opened for reading
(``r``), writing (``w``), or appending (``a``). Must throw an
``IOError`` if the file does not exist or cannot be created, and a
``ValueError`` if the specified value for ``mode`` is not supported. The
additional ``attributes`` arguments specify metadata to be stored in the
file when created. **arfx** will pass all attributes of the channel and
entry (e.g., ``channels``, ``sampling_rate``, ``units``, and
``datatype``) when opening a file for writing. This method may issue a
``ValueError`` if the caller fails to set a required attribute, or
attempts to set an attribute inconsistent with the data format.
Unsupported attributes should be ignored.

``read()``: Reads the contents of the opened file and returns the data
in a format suitable for storage in an ARF file. Specifically, it must
be an acceptable type for the ``arf.entry.add_data()`` method (see
https://github.com/melizalab/arf for documentation).

``write(data)``: Writes data to the file. Must issue an ``IOError`` if
the file is opened in the wrong mode, and ``TypeError`` if the data
format is not correct for the file format.

``timestamp``: A readable property giving the time point of the data.
The value may be a scalar indicating the number of seconds since the
epoch, or a two-element sequence giving the number of seconds and
microseconds since the epoch. If this property is writable it will be
set by **arfx** when writing data.

``sampling_rate``: A property indicating the sampling rate of the data
in the file (or current entry), in units of Hz.

The class may also define the following methods and properties. If any
property is not defined, it is assumed to have the default value defined
below.

``nentries``: A readable property indicating the number of entries in
the file. Default value is 1.

``entry``: A readable and writable integer-valued property corresponding
to the index of the currently active entry in the file. Active means
that the ``read()`` and ``write()`` methods will affect only that entry.
Default is 0, and **arfx** will not attempt to change the property if
``nentries`` is 1.

version information
~~~~~~~~~~~~~~~~~~~

**arfx** uses semantic versioning and is synchronized with the
major/minor version numbers of the arf package specification.
