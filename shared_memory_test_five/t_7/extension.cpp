#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <unordered_map>
#include <functional>
#include <vector>
#include <memory>
#include <numeric>
#include <cstring>
#include "my_library.hpp"
#include "FunctionWrapper.hpp"

// global registry to store functions
static std::unordered_map<std::string, std::unique_ptr<FunctionWrapperBase>> function_registry;

// function to register C++ functions
template<typename Ret, typename... Args>
void register_function(const std::string& name, Ret(*func)(Args...)) {
    function_registry[name] = std::make_unique<FunctionWrapper<Ret, Args...>>(func);
}

// python-callable function to execute registered functions
static PyObject* execute_function(PyObject* self, PyObject* args) {
    const char* func_name;
    PyObject* func_args;

    if (!PyArg_ParseTuple(args, "sO", &func_name, &func_args)) {
        return nullptr;
    }

    auto it = function_registry.find(func_name);
    if (it == function_registry.end()) {
        PyErr_SetString(PyExc_RuntimeError, "Function not found");
        return nullptr;
    }

    return it->second->execute(func_args);
}

static PyObject* addIntsF(PyObject* self, PyObject* args) {
    // Get the number of arguments passed
    Py_ssize_t nargs = PyTuple_Size(args);
    
    // Create a new tuple with the actual arguments
    // Note: we don't need to include the function name since it's hardcoded
    PyObject* func_args = PyTuple_New(nargs);
    if (!func_args) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create arguments tuple");
        return nullptr;
    }
    
    // Copy all arguments to the new tuple
    for (Py_ssize_t i = 0; i < nargs; i++) {
        PyObject* item = PyTuple_GetItem(args, i);
        Py_INCREF(item);  // Increment reference count since PyTuple_SetItem steals the reference
        PyTuple_SetItem(func_args, i, item);
    }
    
    // Find and execute the function
    auto it = function_registry.find("addInts");
    if (it == function_registry.end()) {
        Py_DECREF(func_args);
        PyErr_SetString(PyExc_RuntimeError, "Function not found");
        return nullptr;
    }
    
    // Execute the function and clean up
    PyObject* result = it->second->execute(func_args);
    Py_DECREF(func_args);
    return result;
}

// module method definitions
static PyMethodDef ModuleMethods[] = {
    {"execute_function", execute_function, METH_VARARGS, "Execute a registered function"},
    {"addInts", addIntsF, METH_VARARGS, "add two ints"},
    {nullptr, nullptr, 0, nullptr}
};

// module definition structure
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "fast_cpp_caller",
    "Module for fast C++ function calling",
    -1,
    ModuleMethods
};

// module initialization function
PyMODINIT_FUNC PyInit_fast_cpp_caller(void) {
    import_array();  // initialize NumPy

    PyObject* module = PyModule_Create(&moduledef);
    if (!module) {
        return nullptr;
    }

    // Register all functions during module initialization
    try {
        register_function("addInts", addInts);
        register_function("addArrays", addArrays);
        register_function("addMoreInts", addMoreInts);
        register_function("multiplyArrays", multiplyArrays);
        register_function("scaleArray", scaleArray);
        register_function("dotProduct", dotProduct);
        register_function("expArray", expArray);
        register_function("logArray", logArray);
        register_function("sumArray", sumArray);
        register_function("meanArray", meanArray);
        register_function("maxArray", maxArray);
        register_function("minArray", minArray);
        register_function("linspace", linspace);
        register_function("arange", arange);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, 
            ("Failed to register functions: " + std::string(e.what())).c_str());
        Py_DECREF(module);
        return nullptr;
    }

    return module;
}
