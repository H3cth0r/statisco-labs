#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>

typedef struct {
    PyObject_HEAD
    double *array;
    Py_ssize_t size;
} VectorArrayObject;

static void VectorArray_dealloc(VectorArrayObject *self) 
{
    free(self->array);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *VectorArray_zeros(PyTypeObject *type, PyObject *args)
{
    Py_ssize_t size;
    if (!PyArg_ParseTuple(args, "n", &size)) {
        return NULL;
    }

    VectorArrayObject *self;
    self = (VectorArrayObject *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->size = size;
        self->array = (double *)calloc(size, sizeof(double));
        if (self->array == NULL) {
            Py_DECREF(self);
            PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for the array");
            return NULL;
        }
    }
    return (PyObject *)self;
}

static PyObject *VectorArray_size(VectorArrayObject *self, PyObject *Py_UNUSED(ignored)) 
{
    return PyLong_FromSsize_t(self->size);
}

static PyObject *VectorArray_getitem(VectorArrayObject *self, PyObject *args) 
{
    Py_ssize_t index;

    if (!PyArg_ParseTuple(args, "n", &index)) {
        return NULL;
    }

    if (index < 0 || index >= self->size) {
        PyErr_SetString(PyExc_IndexError, "Index out of bounds");
        return NULL;
    }

    return PyFloat_FromDouble(self->array[index]);
}

static PyObject *VectorArray_setitem(VectorArrayObject *self, PyObject *args)
{
    Py_ssize_t index;
    double value;

    if (!PyArg_ParseTuple(args, "nd", &index, &value)) {
        return NULL;
    }

    if (index < 0 || index >= self->size) {
        PyErr_SetString(PyExc_IndexError, "Index out of bounds");
        return NULL;
    }

    self->array[index] = value;

    Py_RETURN_NONE;
}

static PyMethodDef VectorArray_methods[] = {
    {"size", (PyCFunction)VectorArray_size, METH_NOARGS, "Return the size of the array"},
    {"getItem", (PyCFunction)VectorArray_getitem, METH_VARARGS, "Get item at index"},
    {"setItem", (PyCFunction)VectorArray_setitem, METH_VARARGS, "Set item at index"},
    {NULL}
};

static PyTypeObject VectorArrayType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "vectorarray.VectorArray",
    .tp_basicsize = sizeof(VectorArrayObject),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)VectorArray_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = VectorArray_methods,
};

static PyMethodDef module_methods[] = {
    {"zeros", (PyCFunction)VectorArray_zeros, METH_VARARGS | METH_CLASS, "Create a zero-initialized VectorArray"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef vectorarraymodule = {
    PyModuleDef_HEAD_INIT,
    "vectorarray",
    "VectorArray module",
    -1,
    module_methods
};

PyMODINIT_FUNC PyInit_vectorarray(void)
{
    PyObject *m;
    if (PyType_Ready(&VectorArrayType) < 0)
        return NULL;

    m = PyModule_Create(&vectorarraymodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&VectorArrayType);
    if (PyModule_AddObject(m, "VectorArray", (PyObject *)&VectorArrayType) < 0) {
        Py_DECREF(&VectorArrayType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}

