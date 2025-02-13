#include "function_registry.hpp"
#include <Python.h>

static PyObject* register_functions(PyObject* self, PyObject* args) {
    // This will be implemented in the library that uses this module
    Py_RETURN_NONE;
}

static PyObject* execute_function(PyObject* self, PyObject* args) {
    const char* name;
    PyObject* func_args;
    
    if (!PyArg_ParseTuple(args, "sO", &name, &func_args)) {
        return nullptr;
    }
    
    return FunctionRegistry::getInstance().executeFunction(name, func_args);
}

static PyMethodDef ModuleMethods[] = {
    {"register_functions", register_functions, METH_NOARGS, 
     "Register C++ functions"},
    {"execute_function", execute_function, METH_VARARGS,
     "Execute a registered function"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "fast_functions",
    "Module for fast function execution",
    -1,
    ModuleMethods
};

PyMODINIT_FUNC PyInit_fast_functions(void) {
    import_array();  // Initialize NumPy
    
    PyObject* module = PyModule_Create(&moduledef);
    if (!module) {
        return nullptr;
    }
    
    return module;
}
