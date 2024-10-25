// https://github.com/theashwinabraham/Memory-Allocator
// https://github.com/theashwinabraham/Memory-Allocator/blob/master/Memory.c
#include "Memory.h"
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>
#include <stdint.h>

#define ALIGNMENT 8
#define PAGESIZE ((size_t) sysconf(_SC_PAGESIZE))

static inline size_t roundup(size_t a, size_t b) {
    if (a % b == 0) return a;
    return (1 + a/b)*b;
}

typedef struct memory_header {
    struct memory_header *page_header;
    struct memory_header *prev_block;
    struct memory_header *next_block;
    size_t size;
    size_t total_size;
} memory_header;

static memory_header *head = NULL;
static memory_header *tail = NULL;

void *Malloc(size_t sz) {
    if (sz == 0) return NULL;
    if (head == NULL) {
        head = mmap(NULL, sizeof(memory_header) + sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);

        head->page_header = head;
        head->prev_block  = NULL;
        head->next_block  = NULL;
        head->size        = sz;
        head->total_size  = roundup(sizeof(memory_header) + sz, PAGESIZE);
        tail              = head;
        return ((char *) head) + sizeof(memory_header);
    }
    for (memory_header *ptr = head; ptr != NULL; ptr = ptr->next_block) {
        if (ptr->size == 0 && ptr->total_size >= sizeof(memory_header) + sz) {
            ptr->size = sz;
            return ((char *) ptr) + sizeof(memory_header);
        }
        if (ptr->total_size >= roundup(sizeof(memory_header) + ptr->size, ALIGNMENT) + sizeof(memory_header) + sz) {
            memory_header *n_block = (memory_header *) (((char *) ptr) + roundup(sizeof(memory_header) + ptr->size, ALIGNMENT));
            if(((uintptr_t) n_block) % PAGESIZE >= sizeof(memory_header)) {
                n_block->prev_block   = ptr;
                n_block->next_block   = ptr->next_block;
                if (tail == ptr) tail = n_block;
                else ptr->next_block->prev_block = n_block;
                ptr->next_block       = n_block;
                n_block->page_header  = ptr->page_header;
                n_block->size         = sz;
                n_block->total_size   = ptr->total_size - roundup(sizeof(memory_header) + ptr->size, ALIGNMENT);
                ptr->total_size       = roundup(sizeof(memory_header) + ptr->size, ALIGNMENT);
                return ((char *) n_block) + sizeof(memory_header);
            }
        }
    }
    memory_header *n_block  = mmap(NULL, sizeof(memory_header) + sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    n_block->page_header    = n_block;
    n_block->prev_block     = tail;
    tail->next_block        = n_block;
    tail                    = n_block;
    n_block->total_size     = roundup(sizeof(memory_header) + sz, PAGESIZE);
    n_block->size           = sz;
    return ((char *) n_block) + sizeof(memory_header);
}

void Free(void *ptr) {
    if(ptr == NULL) return;
    memory_header *act_ptr = (memory_header *) (((char*) ptr) - sizeof(memory_header));
    if (act_ptr->prev_block == NULL) {
        if (act_ptr->next_block == NULL) {
            head = NULL;
            tail = NULL;
            munmap(act_ptr->page_header, act_ptr->total_size);
            return;
        }
        act_ptr->size = 0;
    } else {
        if (act_ptr->prev_block->page_header == act_ptr->page_header) {
            act_ptr->prev_block->total_size += act_ptr->total_size;
            act_ptr->prev_block->next_block = act_ptr->next_block;
            if (act_ptr->next_block != NULL) act_ptr->next_block->prev_block = act_ptr->prev_block;
            else tail = act_ptr->prev_block;
        } else {
            act_ptr->size = 0;
        }
    }
    memory_header *pgh = act_ptr->page_header;
    if (pgh->size == 0 && pgh->total_size >= PAGESIZE) {
        if (pgh->total_size % PAGESIZE == 0) {
            if(pgh->prev_block != NULL) pgh->prev_block->next_block = pgh->next_block;
            else head = pgh->next_block;

            if(pgh->next_block != NULL) pgh->next_block->prev_block = pgh->prev_block;
            else tail = pgh->prev_block;
        } else {
            memory_header* n_ptr  = (memory_header *)(((char*) pgh) + (pgh->total_size/PAGESIZE)*PAGESIZE);
            n_ptr->prev_block     = pgh->prev_block;
            n_ptr->next_block     = pgh->next_block;
            n_ptr->size           = 0;
            n_ptr->total_size     = ((char*) pgh->next_block) - ((char*) n_ptr);
            n_ptr->page_header    = pgh;
            for(memory_header *pptr = n_ptr; pptr != NULL && pptr->page_header == pgh; pptr = pptr->next_block) pptr->page_header = n_ptr;

            if(pgh->prev_block != NULL) pgh->prev_block->next_block = n_ptr;
            else head = n_ptr;

            if(pgh->next_block != NULL) pgh->next_block->prev_block = n_ptr;
            else tail = n_ptr;
        }
        munmap(pgh, (pgh->total_size/PAGESIZE)*PAGESIZE);
    }
}

void *Calloc(size_t nmembm size_t szo) {
}
