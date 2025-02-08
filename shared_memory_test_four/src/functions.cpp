// C++ headers first
#include <vector>
#include <cstring>
#include <algorithm>

#include "FunctionRegistry.hpp"

double add_arrays(const double* arr1, const double* arr2, int size) {
    double sum = 0;
    for (int i = 0; i < size; i++) {
        sum += arr1[i] + arr2[i];
    }
    return sum;
}

void multiply_arrays(const double* arr1, const double* arr2, double* result, int size) {
    for(int i = 0; i < size; i++) {
        result[i] = arr1[i] * arr2[i];
    }
}

std::vector<double> subtract_arrays(const std::vector<double>& arr1, const std::vector<double>& arr2) {
    std::vector<double> result(arr1.size());
    for(size_t i = 0; i < arr1.size(); i++) {
        result[i] = arr1[i] - arr2[i];
    }
    return result;
}

extern "C" void register_functions(FunctionRegistry& registry) {
    registry.register_function("add_arrays", add_arrays);
    registry.register_function("multiply_arrays", multiply_arrays);
    registry.register_function("subtract_arrays", subtract_arrays);
}
