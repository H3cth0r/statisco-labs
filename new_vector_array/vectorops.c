#define PY_SSIZE_T_CLEAN
// #include <Python.h>
#include <stdlib.h>
#include <stdio.h>

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

GenericArray* init_array(int dtype, size_t size, void* data, void* data_ptr) {
    // GenericArray *arr = (GenericArray *)malloc(sizeof(GenericArray));
    // GenericArray *arr = (GenericArray *)PyMem_Malloc(sizeof(GenericArray));
    GenericArray *arr = (GenericArray *) data;
    if (arr == NULL) {
      printf("Failed to allocate memory for GenericArray structure\n");
      return NULL;
    }

    // Initialize the structure
    arr->dtype = (Dtype)dtype;
    arr->size = size;

    switch (arr->dtype) {
        case DTYPE_INT:
          // arr->data.int_data    = (int *)malloc(size * sizeof(int));
          arr->data.int_data    = (int *) data_ptr;
          break;
        case DTYPE_FLOAT:
          arr->data.float_data  = (float *)malloc(size * sizeof(float));
          break;
        case DTYPE_DOUBLE:
          arr->data.double_data = (double *)malloc(size * sizeof(double));
          break;
        case DTYPE_CHAR:
          arr->data.char_data   = (char *)malloc(size * sizeof(char));
          break;
        default:
          free(arr);
          printf("Unknown data type\n");
          return NULL;
    }


    if (!arr->data.int_data && !arr->data.float_data && !arr->data.double_data && !arr->data.char_data) {
      free(arr);
      return NULL;
    }
    return arr;
}

void free_array(GenericArray *arr) {
    if (!arr) return;

    switch (arr->dtype) {
      case DTYPE_INT:
        free(arr->data.int_data);
        break;
      case DTYPE_FLOAT:
        free(arr->data.float_data);
        break;
      case DTYPE_DOUBLE:
        free(arr->data.double_data);
        break;
      case DTYPE_CHAR:
        free(arr->data.char_data);
        break;
    }
    free(arr); // Here is where the segmentation is being generated
    return;
}


void set_element(GenericArray *arr, size_t index, void *value) {
    if (index >= arr->size) {
        printf("Index out of bounds\n");
        return;
    }
    switch (arr->dtype) {
        case DTYPE_INT:
            arr->data.int_data[index] = *(int*)value;
            break;
        case DTYPE_FLOAT:
            arr->data.float_data[index] = *(float*)value;
            break;
        case DTYPE_DOUBLE:
            arr->data.double_data[index] = *(double*)value;
            break;
        case DTYPE_CHAR:
            arr->data.char_data[index] = *(char*)value;
            break;
    }
}

void* get_element(GenericArray *arr, size_t index) {
    if (index >= arr->size) {
        printf("Index out of bounds\n");
        return NULL;
    }

    switch (arr->dtype) {
        case DTYPE_INT:
            return &arr->data.int_data[index];
        case DTYPE_FLOAT:
            return &arr->data.float_data[index];
        case DTYPE_DOUBLE:
            return &arr->data.double_data[index];
        case DTYPE_CHAR:
            return &arr->data.char_data[index];
    }
    return NULL;
}

void set_all_elements(GenericArray *arr, void *value) {
    for (size_t i = 0; i < arr->size; i++) {
        set_element(arr, i, value);
    }
}

int test_method() {
  return 24;
}

int main() {
}
