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
      int   *int_data;
      float *float_data;
      double*double_data;
      char  *char_data;
    } data;
    size_t size;
} ArrCo;

ArrCo* init_array(int dtype, size_t size, void* data, void* data_ptr) {
    ArrCo *arr = (ArrCo *) data;
    if (arr == NULL) {
        return NULL;
    }
    
    arr->dtype  = (Dtype)dtype;
    arr->size   = size;

    switch (arr->dtype) {
      case DTYPE_INT:
        arr->data.int_data    = (int *) data_ptr;
        break;
      case DTYPE_FLOAT:
        arr->data.float_data  = (float *) data_ptr;
        break;
      case DTYPE_DOUBLE:
        arr->data.double_data = (double *) data_ptr;
        break;
      case DTYPE_CHAR:
        arr->data.char_data   = (char *) data_ptr;
        break;
      default:
        // free(arr); Not freed since python manages allocation
        return NULL;
    }
    if (!arr->data.int_data && !arr->data.float_data && !arr->data.double_data && !arr->data.char_data ) {
        return NULL;
    }
    return arr;
}

void set_element(ArrCo *arr, size_t index, void *value) {
    if (index >= arr->size) {
        return;
    }
    switch (arr->dtype) {
      case DTYPE_INT:
        arr->data.int_data[index]     = *((int*)value);
        break;
      case DTYPE_FLOAT:
        arr->data.float_data[index]   = *((float*)value);
        break;
      case DTYPE_DOUBLE:
        arr->data.double_data[index]  = *((double*)value);
        break;
      case DTYPE_CHAR:
        arr->data.char_data[index]     = *((char*)value);
        break;
    }
}

void* get_element(ArrCo *arr, size_t index) {
    if (index >= arr->size) {
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

void set_all_elements(ArrCo *arr, void *value) {
    for (size_t i = 0; i < arr->size; i++) {
        switch (arr->dtype) {
          case DTYPE_INT:
            arr->data.int_data[i]     = *((int*)value);
            break;
          case DTYPE_FLOAT:
            arr->data.float_data[i]   = *((float*)value);
            break;
          case DTYPE_DOUBLE:
            arr->data.double_data[i]  = *((double*)value);
            break;
          case DTYPE_CHAR:
            arr->data.char_data[i]     = *((char*)value);
            break;
        }
    }
}

int main() {}
