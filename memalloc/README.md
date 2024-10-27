# Simple Memory Allocator

## Reference
[Rerence article](https://arjunsreedharan.org/post/148675821737/memory-allocators-101-write-a-simple-memory)
[Reference Repo](https://github.com/arjun024/memalloc/tree/master)

## Memory Layour
A process runs within its own virtual address space that's distict from
the virtual address spaces of other processes. This virtual address space
typically comprises of 5 sections:
- Text section: The part that contains the binary instructions to be executed
by the processor.
- Data section: Contains non-zero initialized static data.
- BSS (Block Started by Symbol): Contains zero-initialized static data. Static data
uninitialized in program is initialized 0 and goes here.
- Heap: Contains the dynamically allocated data.
- Stack: Contains your automatic variables, function argumnets, copy of base pointer etc.

![](https://static.tumblr.com/gltvynn/tHIobhrb5/memlayout.jpg)

- The stack and the heap grow in opposite directions.
- Sometimes the bss and the heap are reffered as the "data segment".
- The of which demarcated by a pointer named "program break" or brk.

If we want to allocate more memory in the heap, we need to request the system to
increment brk. Similarly, to release memory we need to request the system to decrement brk.

On Linux(Unix-like systems) we can make use or `sbrk()` system call that lets us manipulate
the program break.
- `sbrk(0)`: gives the current address of program break.
- `sbrk(x)`: with a positive value increments *brk* by *x* bytes, as a result allocating memory.
- `sbrk(-x)`: with negative value decrements *brk* by *x* bytes, as result releasing memory.
- On error, `sbrk()` returns `(void *) -1`.

In the present `sbrk` is not the best options, `mmap` is a better solutions. `sbrk` is not thread
safe, and it can only shrink in LIFO order.

In MacOs, you'll something like this:
```
man 2 sbrk

The brk and sbrk functions are historical curiosities left over from earlier days before the advent of virtual memory management.
```

However, glibc implementation of malloc still uses `sbrk` for allocating memory that's not to big size.

## malloc()
The `malloc(size)` function allocates size bytes of memory and returns a pointer to the allocated memory.
```
void *malloc(size_t size) {
    void *block;
    block = sbrk(size);
    if (block == (void *) -1) return NULL;
    return block;
}
```
- Calling `sbrk()` with the given size.
- On success, size bytes are allocated on the heap.

To then freen the memory, it is necessary to store the size of
the allocated block. Important to understand that the heap memory the
operating system has provided is contiguous. So we can only release
memory which is at the end of the heap. Can't release a block in the middle
of the OS.

To free memory does not means that the memory is being release back to the OS;
it means that we keep the block marked as free. This block marked as free may be 
reused on a later malloc() call. 

Will add a header to every newly allocated memory block:
```
struct header_t {
    size_t size;
    unsigned is_free;
};
```
When program requests for size bytes of memory, we calculate:
```
total_size = header_size + size
sbrk(total_size)
```

Then t the memory blocks will look like this:
```
header + Actual memory block
```

We can't be complete sure the blocks of memory allocated by our malloc 
are contiguous; it can happen that a program has a foreign sbrk or there is a section of memory 
using mmap in between our allocator; therefore we nedd a way to traverse through the blocks
of memory. Then we will use a linked list:
```
struct header_t {
    size_t size;
    unsigned is_free;
    struct header_t *next;
};
```
Wrap the entire header struct in a union along with a stub variable of size 16 bytes. THis makes
the header end up on a memory address aligned to 16 bytes. Recall that the size of a unions is 
the larger of its members. So the unions guarantees that the end of the header is memory allocated.
The end of the header is where the actual memory block begins and therefore the memory provided to
the called by the allocator will be aligned to 16 bytes.
```
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
```

Need a head and tail pointe rto keep track of the list:
```
header_t *head, *tail;
```

To prevent two or more threads from concurrently accessing memory, we will put a basic locking
mechanism in place. We'll have a global lock, and before every action on memory you have to acquire
the lock and once you are done you have to release the lock:
```
pthread_mutex_t global_malloc_lock;
```
Our malloc is now modified to:
```
void *malloc(size_t size) {
    size_t total_size;
    void *block;
    header_t *header;
    if(!size) return NULL;
    pthread_mutex_lock(&global_malloc_lock);
    header = get_free_block(size);
    if (header) {
        header->s.is_free = 0;
        pthread_mutex_unlock(&global_malloc_lock);
        return (void *) (header + 1);
    }
    total_size =  sizeof(header_t) + size;
    block = sbrk(total_size);
    if (block == (void *) -1) {
        pthread_mutex_unlock(&global_malloc_lock),
        return NULL;
    }
    header = block;
    header->s.size = size;
    header->s.is_free = 0;
    header->s.next = NULL;

    if (!head) head = header;
    if (tail) tail->s.next = header;

    tail = header;
    pthread_mutex_unlock(&global_malloc_lock);
    return (void*)(header + 1);
}
header_t *get_free_block(size_t size) {
    header_t *curr = head;
    while (curr) {
        if (curr->s.is_free && curr->s.size >= size) return curr;
        curr = curr->s.next;
    }
    return NULL;
}
```

## Free
First must determine if the block to be freed is at the end of the heap. 
If it is, we can release it to the OS. Otherwise, all we do is mark it
"free" hoping to reuse it later.
```
void free (void *block) {
    header_t *header, *tmp;
    void *programbreak;

    if (!block) return;
    pthread_mutex_lock(&global_malloc_lock);
    header = (header_t *) block - 1;

    programbreak = sbrk(0);
    if ((char *) block + header->s.size == programbreak) {
        if (heade == tail) {
            head = tail = NULL;
        } else {
            tmp = head;
            while (tmp) {
                if (tmp->s.next == tail) {
                    tmp->s.next = NULL;
                    tail = tmp->s.next;
                }
                tmp = tmp->s.next;
            }
        }
        sbrk(0 - sizeof(header_t) - header->s.size);
        pthread_mutex_unlock(&global_malloc_lock);
        return;
    }
    header->s.is_free = 1;
    pthread_mutex_unlock(&global_malloc_lock);
}
```

## calloc
allocates memory for and array of num elements of nsize bytes each 
and returns a pointer to the allocated memory. Addictionally, the 
memory is al set to zeroes.
```
void *calloc(size_t num, size_t nsize) {
    size_t size;
    void *block;
    if (!num || !nsize) return NULL;
    size = num * nsize;
    if (nsize != size / num)  return NULL;
    block = malloc(size);
    if (!block) return NULL;
    memset(block, 0, size);
    return block;
}
```

## realloc()
Changes the size of the given memory block to the size given.
```
void *realloc (void *block, size_t size) {
    header_t *header;
    void *ret;
    if (!block || !size) return malloc(size);
    header = (header_t*)block - 1;
    if (header->s.size >= size) return block;
    ret = malloc(size);
    if (ret) {
        memcpy(ret, block, header->s.size);
        free(block);
    }
    return ret;
}
```
First get the blocks header and see if the block already has the size
to accomodate the requested size. If it does, there's nothing to be done.

If the current block does not have the requested size, then we call malloc 
to get a block of the requested size, and relocate contents to the new
bigger to the bigger block using memcpy. The old memory block is then freed.

## Compiling
Compile as a libary file:
```
gcc -o memalloc.so -fPIC -shared memalloc.c
```
