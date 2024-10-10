#ifndef VECTOROPS_H
#define VECTOROPS_H

#include <stdlib.h>

// Define data types
typedef enum {
    DTYPE_INT,
    DTYPE_FLOAT,
    DTYPE_DOUBLE,
    DTYPE_CHAR
} Dtype;

typedef struct {
    Dtype dtype;
    union {
      int     *int_data;
      float   *float_data;
      double  *double_data;
      char    *char_data;
    } data;
    size_t size;
} GenericArray;

// Declare functions
GenericArray* init_array(int dtype, size_t size);
void free_array(GenericArray *arr);
void set_element(GenericArray *arr, size_t index, void *value);
void* get_element(GenericArray *arr, size_t index);
void set_all_elements(GenericArray *arr, void *value);
int test_method();

#endif
