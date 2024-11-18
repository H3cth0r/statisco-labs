// fast_array_opt.h
#ifndef FAST_ARRAY_OPT_H
#define FAST_ARRAY_OPT_H

#include <stdlib.h>
#include <string.h>
#include <immintrin.h>  // For AVX instructions
#include <stdio.h>

typedef struct {
    size_t size;
    void* data;
    size_t itemsize;
} FastArray;

// Function declarations
FastArray* create_array(size_t size, size_t itemsize);
void destroy_array(FastArray* arr);
void fill_zeros_optimized(FastArray* arr);
void fill_range_optimized(FastArray* arr);
long long sum_array_optimized(FastArray* arr);

#endif
