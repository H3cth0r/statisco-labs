// extension.cpp
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

// Function wrapper base class
class FunctionWrapperBase {
public:
    virtual PyObject* execute(PyObject* args) = 0;
    virtual ~FunctionWrapperBase() = default;
};

// Template class for handling different function signatures
template<typename Ret, typename... Args>
class FunctionWrapper : public FunctionWrapperBase {
private:
    std::function<Ret(Args...)> func;
    
    // Helper to convert Python objects to C++ types
    template<typename T>
    T convert_arg(PyObject* obj) {
        using U = std::decay_t<T>;
        if constexpr (std::is_same_v<U, int>) {
            return PyLong_AsLong(obj);
        } else if constexpr (std::is_same_v<U, double>) {
            return PyFloat_AsDouble(obj);
        } else if constexpr (std::is_same_v<U, std::vector<double>>) {
            // (Conversion for numpy array to std::vector<double> remains the same)
            if (!PyArray_Check(obj)) {
                throw std::runtime_error("Expected numpy array");
            }
            
            PyArrayObject* arr_cont = reinterpret_cast<PyArrayObject*>(
                PyArray_ContiguousFromAny(obj, NPY_DOUBLE, 1, 1));
            
            if (!arr_cont) {
                throw std::runtime_error("Could not convert array to contiguous double array");
            }
            
            npy_intp size = PyArray_SIZE(arr_cont);
            double* data = static_cast<double*>(PyArray_DATA(arr_cont));
            std::vector<double> result(data, data + size);
            
            Py_DECREF(arr_cont);
            return result;
        }
        throw std::runtime_error("Unsupported type conversion");
    }

    // Helper to convert C++ return types to Python objects
    template<typename T>
    PyObject* convert_return(const T& value) {
        if constexpr (std::is_same_v<T, int>) {
            return PyLong_FromLong(value);
        } else if constexpr (std::is_same_v<T, double>) {
            return PyFloat_FromDouble(value);
        } else if constexpr (std::is_same_v<T, std::vector<double>>) {
            npy_intp dims[] = {static_cast<npy_intp>(value.size())};
            PyObject* arr = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
            if (!arr) {
                throw std::runtime_error("Failed to create numpy array");
            }
            memcpy(PyArray_DATA(reinterpret_cast<PyArrayObject*>(arr)), 
                   value.data(), value.size() * sizeof(double));
            return arr;
        }
        throw std::runtime_error("Unsupported return type conversion");
    }

    // Helper to get argument at index
    template<size_t I>
    auto get_arg(PyObject* args) {
        using ArgType = typename std::tuple_element<I, std::tuple<Args...>>::type;
        // Remove both reference and const qualifiers.
        return convert_arg<std::remove_cv_t<std::remove_reference_t<ArgType>>>(PyTuple_GetItem(args, I));
    }

    // Helper to build argument tuple
    template<size_t... I>
    auto build_args(PyObject* args, std::index_sequence<I...>) {
        return std::make_tuple(get_arg<I>(args)...);
    }

public:
    FunctionWrapper(std::function<Ret(Args...)> f) : func(f) {}

    PyObject* execute(PyObject* args) override {
        if (!PyTuple_Check(args)) {
            PyErr_SetString(PyExc_TypeError, "Arguments must be a tuple");
            return nullptr;
        }

        if (PyTuple_Size(args) != sizeof...(Args)) {
            PyErr_SetString(PyExc_TypeError, "Wrong number of arguments");
            return nullptr;
        }

        try {
            auto tuple_args = build_args(args, std::index_sequence_for<Args...>{});
            if constexpr (std::is_void_v<Ret>) {
                std::apply(func, tuple_args);
                Py_RETURN_NONE;
            } else {
                auto result = std::apply(func, tuple_args);
                return convert_return(result);
            }
        } catch (const std::exception& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
            return nullptr;
        }
    }
};

// Global registry to store functions
static std::unordered_map<std::string, std::unique_ptr<FunctionWrapperBase>> function_registry;

// Function to register C++ functions
template<typename Ret, typename... Args>
void register_function(const std::string& name, Ret(*func)(Args...)) {
    function_registry[name] = std::make_unique<FunctionWrapper<Ret, Args...>>(func);
}

// Python-callable function to execute registered functions
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

// Function to register all functions from my_library.hpp
static PyObject* register_functions(PyObject* self, PyObject* args) {
    // Register your functions here
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
    Py_RETURN_NONE;
}

// Module method definitions
static PyMethodDef ModuleMethods[] = {
    {"execute_function", execute_function, METH_VARARGS, "Execute a registered function"},
    {"register_functions", register_functions, METH_NOARGS, "Register all available functions"},
    {nullptr, nullptr, 0, nullptr}
};

// Module definition structure
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "fast_cpp_caller",
    "Module for fast C++ function calling",
    -1,
    ModuleMethods
};

// Module initialization function
PyMODINIT_FUNC PyInit_fast_cpp_caller(void) {
    import_array();  // Initialize NumPy

    PyObject* module = PyModule_Create(&moduledef);
    if (!module) {
        return nullptr;
    }

    return module;
}
