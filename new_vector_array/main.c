#include <stdio.h>
#include "vectorops.h"

int main() {
    // Test the init_array function for integers
    size_t array_size = 5;
    GenericArray *intArray = init_array(DTYPE_INT, array_size);

    if (!intArray) {
        printf("Failed to initialize array.\n");
        return 1;
    }

    // Set and get elements in the array
    int value = 42;
    set_element(intArray, 0, &value);

    int *retrievedValue = (int*)get_element(intArray, 0);
    if (retrievedValue) {
        printf("First element in intArray: %d\n", *retrievedValue);
    }

    // Set all elements in the array to a single value
    value = 99;
    set_all_elements(intArray, &value);

    for (size_t i = 0; i < array_size; i++) {
        printf("intArray[%zu] = %d\n", i, intArray->data.int_data[i]);
    }

    // Free the array
    free_array(intArray);

    // Test other data types
    GenericArray *floatArray = init_array(DTYPE_FLOAT, array_size);
    if (floatArray) {
        float floatValue = 3.14;
        set_element(floatArray, 0, &floatValue);

        float *retrievedFloat = (float*)get_element(floatArray, 0);
        if (retrievedFloat) {
            printf("First element in floatArray: %f\n", *retrievedFloat);
        }

        free_array(floatArray);
    }

    // Test the test_method function
    printf("Test method returned: %d\n", test_method());

    return 0;
}
