#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    PyObject_HEAD;
    PyObject *cache;
} MallocAllocator;
