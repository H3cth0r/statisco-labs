#include "function_registry.hpp"

template<>
struct TypeConverter<int> {
    static int fromPyObject(PyObject* obj) {
        return PyLong_AsLong(obj);
    }
    
    static PyObject* toPyObject(const int& value) {
        return PyLong_FromLong(value);
    }
};

template<>
struct TypeConverter<NumpyArrayWrapper*> {
    static NumpyArrayWrapper* fromPyObject(PyObject* obj) {
        return new NumpyArrayWrapper((PyArrayObject*)obj);
    }
    
    static PyObject* toPyObject(NumpyArrayWrapper* value) {
        return value->getPyObject();
    }
};

NumpyArrayWrapper::NumpyArrayWrapper(PyArrayObject* arr, bool takeOwnership) 
    : array(arr), ownsData(takeOwnership) {
    data = PyArray_DATA(arr);
}

NumpyArrayWrapper::~NumpyArrayWrapper() {
    if (ownsData) {
        Py_XDECREF(array);
    }
}

npy_intp* NumpyArrayWrapper::getDimensions() {
    return PyArray_DIMS(array);
}

int NumpyArrayWrapper::getNDim() {
    return PyArray_NDIM(array);
}

PyObject* NumpyArrayWrapper::getPyObject() {
    Py_INCREF(array);
    return (PyObject*)array;
}

NumpyArrayWrapper* NumpyArrayWrapper::createEmpty(int ndim, npy_intp* dims, NPY_TYPES dtype) {
    PyArrayObject* arr = (PyArrayObject*)PyArray_SimpleNew(ndim, dims, dtype);
    return new NumpyArrayWrapper(arr, true);
}

FunctionRegistry& FunctionRegistry::getInstance() {
    static FunctionRegistry instance;
    return instance;
}

PyObject* FunctionRegistry::executeFunction(const char* name, PyObject* args) {
    auto it = functions.find(name);
    if (it == functions.end()) {
        PyErr_SetString(PyExc_RuntimeError, "Function not found");
        return nullptr;
    }
    return it->second->execute(args);
}
