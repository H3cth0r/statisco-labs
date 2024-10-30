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
}
