### Adaptation of Max Imakaev's fast txt matrix writer.

cimport cython
import os
import subprocess

from libc.stdlib cimport malloc, free
from libc.string cimport strcpy, strlen

import numpy as np
cimport numpy as np

cdef extern from "stdio.h":
    int sprintf(char *str, char *format, ...)

def commandExists(command):
    """
    Checks if the bash command exists.
    """
    command = command.split()[0]
    if subprocess.call(['which', command]) != 0:
        return False
    return True

def gzipWriter(filepath):
    """
    Creates a writing process with gzip or parallel gzip (pigz) attached to it.
    """
    filepath = os.path.abspath(filepath)
    with open(filepath, 'wb') as outFile:
        if commandExists("pigz"):
            writer = ["pigz", "-c", "-9"]
        else:
            writer = ["gzip", "-c", "-2"]

        pwrite = subprocess.Popen(
            writer,
            stdin=subprocess.PIPE,
            stdout=outFile,
            shell=False,
            bufsize=-1)
    return pwrite

def empty_func():
    return None

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)

def array2txt(
    mat,
    out,
    format_string=b'%.4lf',
    sep=b'\t',
    newline=b'\n',
    header=None,
    row_headers=None,
    max_element_len=100):
    """
    Dump a 2d array into a text file, optionally gzipped.
    This implementation if faster than the np.savetxt and it allows the user
    to provide column/row headers.

    Parameters
    ----------
    mat : a 2D numpy array of a list of lists of numbers (float/integer)

    out : str or file object
        Either:
        -- a path to the output file. If ends with .gz the output is gzipped
        -- a file object
        -- a stdin of a Popen object
        TIP: when using files/stdin do not forget to flush()/communicate().

    format_string : bytes, optional
        A printf-style formatting string to specify the coversion of
        the elements of the matrix into strings.

    sep : bytes, optional
        The column separator.

    newline : bytes, optional
        The newline separator.

    header : bytes, optional
        A header to prepend to the output file, is separated from the main table
        by a `newline`.

    row_headers : a list of bytes, optional
        Row headers to prepend to the output file, one per each row in `mat`.

    max_element_len : int
            The maximal length of the string representation of a matrix element,
        produced by sprintf(`format_string`). Used to preallocate memory.
    """


    cdef int N = len(mat)
    cdef int M = len(mat[0])

    if issubclass(type(out), str) or issubclass(type(out), bytearray):
        if out.endswith('.gz'):
            writer = gzipWriter(out)
            out_pipe = writer.stdin
            close_out_func = writer.communicate
        else:
            writer = open(out, 'wb')
            out_pipe = writer
            close_out_func = writer.flush
    elif hasattr(out, 'write'):
        out_pipe = out
        close_out_func = empty_func
    else:
        raise Exception('`out` must be either a file path or a file handle/stream')

    cdef np.ndarray[np.double_t, ndim=2] mat_ndarray = np.array(mat, dtype=np.double, order="C")

    cdef char* newline_cstr = newline
    cdef char* sep_cstr = sep
    cdef char* next_row_header
    cdef char* s_start
    cdef char* s_cur

    cdef int max_header_len = 0
    if row_headers is not None:
        max_header_len = max([len(row_header) for row_header in row_headers])

    s_start = <char *>malloc((max_element_len * M  + max_header_len) * sizeof(char))

    cdef double element
    cdef char* curStringTemplate
    template = b''.join([format_string, sep])
    curStringTemplate = template

    if header is not None:
        out_pipe.write(header)
        out_pipe.write(newline_cstr)

    cdef int i,j
    for i in xrange(N):
        s_cur = s_start
        if row_headers is not None:

            next_row_header = row_headers[i]
            s_cur = strcpy(s_cur, next_row_header)
            s_cur += sizeof(char) * strlen(next_row_header)

            s_cur = strcpy(s_cur, sep_cstr)
            s_cur += sizeof(char) * strlen(sep_cstr)

        for j in xrange(M):
            element = mat_ndarray[i,j]
            s_cur = s_cur + sprintf(s_cur, curStringTemplate, element)

        s_cur = strcpy(s_cur, newline_cstr)
        s_cur += sizeof(char) * strlen(newline_cstr)

        out_pipe.write(s_start)
    free(s_start)

    close_out_func()
