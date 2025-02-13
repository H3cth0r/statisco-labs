#include "my_library.hpp"
#include "function_registry.hpp"
#include <stdexcept>

int addInts(int a, int b) {
    return a + b;
}

int addMoreInts(int a, int b, int c) {
    return a + b + c;
}

NumpyArrayWrapper* addArrays(NumpyArrayWrapper* a, NumpyArrayWrapper* b) {
    // Check dimensions
    if (a->getNDim() != 1 || b->getNDim() != 1) {
        throw std::runtime_error("Arrays must be 1-dimensional");
    }
    
    npy_intp* dims_a = a->getDimensions();
    npy_intp* dims_b = b->getDimensions();
    
    if (dims_a[0] != dims_b[0]) {
        throw std::runtime_error("Arrays must have the same size");
    }
    
    // Create output array
    NumpyArrayWrapper* result = NumpyArrayWrapper::createEmpty(1, dims_a, NPY_INT32);
    
    // Perform addition
    int* data_a = a->getData<int>();
    int* data_b = b->getData<int>();
    int* data_result = result->getData<int>();
    
    for (npy_intp i = 0; i < dims_a[0]; i++) {
        data_result[i] = data_a[i] + data_b[i];
    }
    
    return result;
}

PyObject* register_functions(PyObject* self, PyObject* args) {
    auto& registry = FunctionRegistry::getInstance();
    
    try {
        registry.registerFunction("addInts", addInts);
        registry.registerFunction("addMoreInts", addMoreInts);
        registry.registerFunction("addArrays", addArrays);
    }
    catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
    
    Py_RETURN_NONE;
}

// Template instantiation for our function types
template void FunctionRegistry::registerFunction<int, int, int>(
    const std::string&, int(*)(int, int));
template void FunctionRegistry::registerFunction<int, int, int, int>(
    const std::string&, int(*)(int, int, int));
template void FunctionRegistry::registerFunction<NumpyArrayWrapper*, NumpyArrayWrapper*, NumpyArrayWrapper*>(
    const std::string&, NumpyArrayWrapper*(*)(NumpyArrayWrapper*, NumpyArrayWrapper*));
