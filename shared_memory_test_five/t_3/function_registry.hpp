#pragma once
#include <Python.h>
#include <numpy/arrayobject.h>
#include <unordered_map>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// Forward declaration
class FunctionWrapper;

class FunctionRegistry {
private:
    std::unordered_map<std::string, std::unique_ptr<FunctionWrapper>> functions;

public:
    template<typename Ret, typename... Args>
    void registerFunction(const std::string& name, Ret(*func)(Args...));
    
    PyObject* executeFunction(const char* name, PyObject* args);
    static FunctionRegistry& getInstance();

private:
    FunctionRegistry() = default;
    FunctionRegistry(const FunctionRegistry&) = delete;
    FunctionRegistry& operator=(const FunctionRegistry&) = delete;
};

class FunctionWrapper {
public:
    virtual PyObject* execute(PyObject* args) = 0;
    virtual ~FunctionWrapper() = default;
};

template<typename T>
struct TypeConverter {
    static T fromPyObject(PyObject* obj);
    static PyObject* toPyObject(const T& value);
};

// Specialization for numpy arrays
class NumpyArrayWrapper {
private:
    PyArrayObject* array;
    void* data;
    bool ownsData;

public:
    NumpyArrayWrapper(PyArrayObject* arr, bool takeOwnership = false);
    ~NumpyArrayWrapper();
    
    template<typename T>
    T* getData() { return static_cast<T*>(data); }
    
    npy_intp* getDimensions();
    int getNDim();
    PyObject* getPyObject();
    
    static NumpyArrayWrapper* createEmpty(int ndim, npy_intp* dims, NPY_TYPES dtype);
};

template<typename Ret, typename... Args>
class TypedFunctionWrapper : public FunctionWrapper {
private:
    std::function<Ret(Args...)> func;
    
    template<size_t... I>
    auto callFunction(PyObject* args, std::index_sequence<I...>) {
        std::tuple<typename std::decay<Args>::type...> converted;
        PyArg_UnpackTuple(args, "", sizeof...(Args), sizeof...(Args), 
            &std::get<I>(converted)...);
        return func(TypeConverter<typename std::decay<Args>::type>::fromPyObject(
            std::get<I>(converted))...);
    }

public:
    TypedFunctionWrapper(std::function<Ret(Args...)> f) : func(f) {}
    
    PyObject* execute(PyObject* args) override {
        try {
            if constexpr(std::is_void_v<Ret>) {
                callFunction(args, std::index_sequence_for<Args...>{});
                Py_RETURN_NONE;
            } else {
                auto result = callFunction(args, std::index_sequence_for<Args...>{});
                return TypeConverter<Ret>::toPyObject(result);
            }
        } catch (const std::exception& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
            return nullptr;
        }
    }
};
