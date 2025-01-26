#define PY_SSIZE_T_CLEAN
#include <Python.h>

static PyObject* inspect_interpreter_state(PyObject *self, PyObject *args) {
    // Use public API to get the modules dictionary
    PyObject *modules_dict = PyImport_GetModuleDict();
    if (modules_dict == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to retrieve modules dictionary");
        return NULL;
    }

    if (PyDict_Check(modules_dict)) {
        printf("Modules:\n");

        PyObject *key, *value;
        Py_ssize_t pos = 0;

        while (PyDict_Next(modules_dict, &pos, &key, &value)) {
            PyObject *key_repr = PyObject_Repr(key);
            PyObject *value_repr = PyObject_Repr(value);
            const char *key_str = PyUnicode_AsUTF8(key_repr);
            const char *value_str = PyUnicode_AsUTF8(value_repr);

            printf("  %s: %s\n", key_str, value_str);

            Py_XDECREF(key_repr);
            Py_XDECREF(value_repr);
        }
    } else {
        printf("Modules dictionary is not a dict.\n");
    }

    Py_RETURN_NONE;
}

static PyMethodDef InterpstateMethods[] = {
    {"inspect", inspect_interpreter_state, METH_NOARGS, "Inspect the modules dictionary."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef interpstatemodule = {
    PyModuleDef_HEAD_INIT,
    "interpstate",
    NULL,
    -1,
    InterpstateMethods
};

PyMODINIT_FUNC PyInit_interpstate(void) {
    return PyModule_Create(&interpstatemodule);
}
