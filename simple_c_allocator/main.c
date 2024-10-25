#include "heap.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
    int i;

    heap_t *heap = malloc(sizeof(heap_t));
    memset(heap, 0, sizeof(heap_t));

    void *region = malloc(HEAP_INIT_SIZE);
    memset(heap, 0, HEAP_INIT_SIZE);

    for (i = 0; i < BIN_COUNT; i++) {
    }
}
