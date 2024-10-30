#ifndef MEMALLOC_H
#define MEMALLOC_H

#include <stddef.h>
#include <pthread.h>

extern pthread_mutex_t global_malloc_lock;

void *malloc(size_t size);
void *calloc(size_t num, size_t n_size);
void *realloc(void *block, size_t size);
void free(void *block);
void print_mem_list(void);

#endif
