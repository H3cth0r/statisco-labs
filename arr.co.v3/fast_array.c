// fast_array_opt.c
#include "fast_array.h"

// Align to 32-byte boundary for AVX
#define ALIGN_SIZE 32

FastArray* create_array(size_t size, size_t itemsize) {
    FastArray* arr = (FastArray*)malloc(sizeof(FastArray));
    if (!arr) return NULL;
    
    // Align memory allocation for SIMD operations
    size_t total_size = size * itemsize;
    arr->data = aligned_alloc(ALIGN_SIZE, ((total_size + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE);
    if (!arr->data) {
        free(arr);
        return NULL;
    }
    
    arr->size = size;
    arr->itemsize = itemsize;
    return arr;
}

void destroy_array(FastArray* arr) {
    if (!arr) return;
    free(arr->data);
    free(arr);
}

void fill_zeros_optimized(FastArray* arr) {
    if (!arr || !arr->data) return;
    
    size_t total_size = arr->size * arr->itemsize;
    
    // Use AVX instructions for faster zeroing
    __m256i zero = _mm256_setzero_si256();
    size_t vec_size = sizeof(__m256i);
    size_t vec_count = total_size / vec_size;
    
    __m256i* vec_ptr = (__m256i*)arr->data;
    for (size_t i = 0; i < vec_count; i++) {
        _mm256_store_si256(vec_ptr + i, zero);
    }
    
    // Handle remaining bytes
    size_t remaining = total_size % vec_size;
    if (remaining > 0) {
        memset((char*)arr->data + (total_size - remaining), 0, remaining);
    }
}

void fill_range_optimized(FastArray* arr) {
    if (!arr || !arr->data) return;
    
    // Use AVX2 for parallel processing
    __m256i increment = _mm256_set_epi32(7,6,5,4,3,2,1,0);
    __m256i add_eight = _mm256_set1_epi32(8);
    
    int* data = (int*)arr->data;
    size_t vec_count = arr->size / 8;
    
    __m256i current = increment;
    for (size_t i = 0; i < vec_count; i++) {
        _mm256_store_si256((__m256i*)(data + i * 8), current);
        current = _mm256_add_epi32(current, add_eight);
    }
    
    // Handle remaining elements
    for (size_t i = vec_count * 8; i < arr->size; i++) {
        data[i] = i;
    }
}

long long sum_array_optimized(FastArray* arr) {
    if (!arr || !arr->data) return 0;
    
    int* data = (int*)arr->data;
    __m256i sum_vec = _mm256_setzero_si256();
    size_t vec_count = arr->size / 8;
    
    // Process 8 integers at a time using AVX2
    for (size_t i = 0; i < vec_count; i++) {
        __m256i current = _mm256_load_si256((__m256i*)(data + i * 8));
        sum_vec = _mm256_add_epi32(sum_vec, current);
    }
    
    // Extract and sum the individual elements from the vector
    int sum_array[8];
    _mm256_store_si256((__m256i*)sum_array, sum_vec);
    long long total = 0;
    for (int i = 0; i < 8; i++) {
        total += sum_array[i];
    }
    
    // Handle remaining elements
    for (size_t i = vec_count * 8; i < arr->size; i++) {
        total += data[i];
    }
    
    return total;
}
