#include <stdexcept>

// function wrapper base class
class FunctionWrapperBase {
public:
    virtual PyObject* execute(PyObject* args) = 0;
    virtual ~FunctionWrapperBase() = default;
};

// template class for handling different function signatures
template<typename Ret, typename... Args>
class FunctionWrapper : public FunctionWrapperBase {
private:
    std::function<Ret(Args...)> func;
    
    // helper to convert Python objects to C++ types
    template<typename T>
    T convert_arg(PyObject* obj) {
        using U = std::decay_t<T>;
        if constexpr (std::is_same_v<U, int>) {
            return PyLong_AsLong(obj);
        } else if constexpr (std::is_same_v<U, double>) {
            return PyFloat_AsDouble(obj);
        } else if constexpr (std::is_same_v<U, std::vector<double>>) {
            // (conversion for numpy array to std::vector<double> remains the same)
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

    // helper to convert C++ return types to Python objects
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

    // helper to get argument at index
    template<size_t I>
    auto get_arg(PyObject* args) {
        using ArgType = typename std::tuple_element<I, std::tuple<Args...>>::type;
        // Remove both reference and const qualifiers.
        return convert_arg<std::remove_cv_t<std::remove_reference_t<ArgType>>>(PyTuple_GetItem(args, I));
    }

    // helper to build argument tuple
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
