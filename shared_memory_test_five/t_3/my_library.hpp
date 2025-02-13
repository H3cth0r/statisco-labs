#pragma once
#include "function_registry.hpp"
#include <Python.h>

// Basic integer operations
int addInts(int a, int b);
int addMoreInts(int a, int b, int c);

// Array operations using NumpyArrayWrapper
NumpyArrayWrapper* addArrays(NumpyArrayWrapper* a, NumpyArrayWrapper* b);

// Python registration function
PyObject* register_functions(PyObject* self, PyObject* args);
