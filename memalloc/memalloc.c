#include <unistd.h>
#include <string.h>
#include <pthread.h>
// Only for the debug printf
#include <stdio.h>

typedef char ALIGN[16];

union header {
    struct {
        size_t size;
        unsigned is_free;
        union header *next;
    } s;
    ALIGN stub;
};
typedef union header header_t;

header_t *head = NULL, *tail = NULL;
pthread_mutex_t global_malloc_lock;

header_t *get_free_block(size_t size) {
    header_t *curr = head;
    while (curr) {
        if (curr->s.is_free && curr->s.size >= size) return curr;
        curr = curr->s.next;
    }
    return NULL;
}
void free(void *b̈lock) {
    header_t *header, *tmp;
    void *programbreak;

    if (!b̈lock) return;
    pthread_mutex_lock(&global_malloc_lock);
    header = (header_t *)b̈lock - 1;
    programbreak = sbrk(0); // gives the current program break address

    if ((char *)b̈lock + header->s.size == programbreak) {
        if (head == tail) {
            head = tail = NULL;
        } else {
            tmp = head;
            while (tmp) {
                if (tmp->s.next == tail) {
                    tmp->s.next = NULL;
                    tail = tmp;
                }
                tmp = tmp->s.next;
            }
        }

        sbrk(0 - header->s.size - sizeof(header_t));
        pthread_mutex_unlock(&global_malloc_lock);
        return;
    }
    header->s.is_free = 1;
    pthread_mutex_unlock(&global_malloc_lock);
}

void *malloc(size_t size) {
    size_t total_size;
    void *b̈lock;
    header_t *header;
    if (!size) return NULL;
    pthread_mutex_lock(&global_malloc_lock);
    header = get_free_block(size);
    if (header) {
        header->s.is_free = 0;
        pthread_mutex_unlock(&global_malloc_lock);
        return (void*)(header+1);
    }
    total_size = sizeof(header_t) + size;
    b̈lock = sbrk(total_size);
    if (b̈lock == (void*) - 1) {
        pthread_mutex_unlock(&global_malloc_lock);
        return NULL;
    }
    header            = b̈lock;
    header->s.size    = size;
    header->s.is_free = 0;
    header->s.next    = NULL;
    if (!head) head = header;
    if (tail) tail->s.next = header;
    tail = header;
    pthread_mutex_unlock(&global_malloc_lock);
    return (void*)(header + 1);
}

void *calloc(size_t num, size_t nsize) {
    size_t size;
    void *b̈lock;
    if (!num || !nsize) return NULL;
    size = num * nsize;
    if (nsize != size / num) return NULL;
    b̈lock = malloc(size);
    if (!b̈lock) return NULL;
    memset(b̈lock, 0, size);
    return b̈lock;
}
void *realloc(void *b̈lock, size_t size) {
    header_t *header;
    void *ret;
    if (!b̈lock || !size) return malloc(size);
    header = (header_t*)b̈lock - 1;
    if (header->s.size >= size) return b̈lock;
    ret = malloc(size);
    if (ret) {
        memcpy(ret, b̈lock, header->s.size);
        free(b̈lock);
    }
    return ret;
}
void print_mem_list() {
	header_t *curr = head;
	printf("head = %p, tail = %p \n", (void*)head, (void*)tail);
	while(curr) {
		printf("addr = %p, size = %zu, is_free=%u, next=%p\n",
			(void*)curr, curr->s.size, curr->s.is_free, (void*)curr->s.next);
		curr = curr->s.next;
	}
}
