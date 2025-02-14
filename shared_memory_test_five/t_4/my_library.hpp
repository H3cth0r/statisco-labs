// my_library.hpp
#pragma once
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <cmath>

// Basic integer operations
inline int addInts(int a, int b) {
    return a + b;
}

inline int addMoreInts(int a, int b, int c) {
    return a + b + c;
}

// Array operations using std::vector
inline std::vector<double> addArrays(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Arrays must have the same size");
    }
    
    std::vector<double> result(a.size());
    std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::plus<double>());
    return result;
}

inline std::vector<double> multiplyArrays(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Arrays must have the same size");
    }
    
    std::vector<double> result(a.size());
    std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::multiplies<double>());
    return result;
}

inline std::vector<double> scaleArray(const std::vector<double>& arr, double scalar) {
    std::vector<double> result(arr.size());
    std::transform(arr.begin(), arr.end(), result.begin(),
                  [scalar](double x) { return x * scalar; });
    return result;
}

inline double dotProduct(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Arrays must have the same size");
    }
    
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

// Element-wise operations
inline std::vector<double> expArray(const std::vector<double>& arr) {
    std::vector<double> result(arr.size());
    std::transform(arr.begin(), arr.end(), result.begin(),
                  [](double x) { return std::exp(x); });
    return result;
}

inline std::vector<double> logArray(const std::vector<double>& arr) {
    std::vector<double> result(arr.size());
    std::transform(arr.begin(), arr.end(), result.begin(),
                  [](double x) { 
                      if (x <= 0) throw std::runtime_error("Log of non-positive number");
                      return std::log(x); 
                  });
    return result;
}

// Reduction operations
inline double sumArray(const std::vector<double>& arr) {
    return std::accumulate(arr.begin(), arr.end(), 0.0);
}

inline double meanArray(const std::vector<double>& arr) {
    if (arr.empty()) {
        throw std::runtime_error("Cannot compute mean of empty array");
    }
    return sumArray(arr) / arr.size();
}

inline double maxArray(const std::vector<double>& arr) {
    if (arr.empty()) {
        throw std::runtime_error("Cannot compute max of empty array");
    }
    return *std::max_element(arr.begin(), arr.end());
}

inline double minArray(const std::vector<double>& arr) {
    if (arr.empty()) {
        throw std::runtime_error("Cannot compute min of empty array");
    }
    return *std::min_element(arr.begin(), arr.end());
}

// Array creation utilities
inline std::vector<double> linspace(double start, double end, size_t num) {
    if (num < 2) {
        throw std::runtime_error("Number of points must be at least 2");
    }
    
    std::vector<double> result(num);
    double step = (end - start) / (num - 1);
    
    for (size_t i = 0; i < num; ++i) {
        result[i] = start + step * i;
    }
    
    return result;
}

inline std::vector<double> arange(double start, double end, double step = 1.0) {
    if (step == 0.0) {
        throw std::runtime_error("Step size cannot be zero");
    }
    
    size_t num = static_cast<size_t>((end - start) / step);
    std::vector<double> result(num);
    
    for (size_t i = 0; i < num; ++i) {
        result[i] = start + step * i;
    }
    
    return result;
}
