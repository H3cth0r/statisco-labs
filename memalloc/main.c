#include <stdio.h>
#include "memalloc.h"

int main() {
    pthread_mutex_init(&global_malloc_lock, NULL);

    // Test 1: Basic allocation and deallocation
    printf("Test 1: Basic malloc and free\n");
    int *arr = (int *)malloc(5 * sizeof(int));
    if (arr) {
        for (int i = 0; i < 5; i++) {
            arr[i] = i + 1;
        }
        printf("Allocated array and filled with values: \n");
        for (int i = 0; i < 5; i++) {
            printf("%d ", arr[i]);
        }
        printf("\n");
        print_mem_list();
        free(arr);
        printf("After freeing:\n");
        print_mem_list();
    }

    // Test 2: calloc
    printf("\nTest 2: calloc\n");
    int *carr = (int *)calloc(5, sizeof(int));
    if (carr) {
        printf("Calloc'd array (should be all zeros):\n");
        for (int i = 0; i < 5; i++) {
            printf("%d",  carr[i]);
        }
        printf("\n");
        print_mem_list();
        free(carr);
        printf("After freeing:\n");
        print_mem_list();
    }

    // Test 3: realloc
    printf("\nTest 3: realloc\n");
    int *rarr = (int *)malloc(3 * sizeof(int));
    if (rarr) {
        for (int i = 0; i < 3; i++) {
            rarr[i] = i + 1;
        }
        printf("Original array:\n");
        for (int i = 0; i < 3; i++) {
            printf("%d", rarr[i]);
        }
        printf("\n");

        rarr = (int *)realloc(rarr, 5 * sizeof(int));
        if (rarr) {
            rarr[3] = 4;
            rarr[4] = 5;
            printf("Reallocated array:\n");
            for (int i = 0; i < 5; i++) {
                printf("%d", rarr[i]);
            }
            printf("\n");
            print_mem_list();
            free(rarr);
            printf("After freeing:\n");
            print_mem_list();
        }
    }

    pthread_mutex_destroy(&global_malloc_lock);
    return 0;
}
