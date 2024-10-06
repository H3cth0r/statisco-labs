#define PY_SSIZE_T_CLEAN
#include <Python.h>

int addition(int a, int b) {
  return a + b;
}

PyObject *add(PyObject *self, PyObject *args)
{
    int a, b;
    if (!PyArg_ParseTuple(args, "ii", &a, &b)) {
        return NULL;
    }
    return PyLong_FromLong(a + b);
}

PyObject *subtract(PyObject* self, PyObject* args) 
{
    int a, b;
    if (!PyArg_ParseTuple(args, "ii", &a, &b)) {
        return NULL;
    }

    return PyLong_FromLong(a - b);
}
