#include <stdio.h>
#include <stdlib.h>

// Function to sum two integers
int sum(int a, int b) {
    return a + b;
}

// Function to create an array of zeros
int* create_array(int size) {
    int* array = (int*)malloc(size * sizeof(int)); // Allocate memory for the array
    if (!array) return NULL; // Return NULL if memory allocation fails
    for (int i = 0; i < size; i++) {
        array[i] = 0; // Initialize all elements to zero
    }
    return array; // Return the pointer to the array
}
