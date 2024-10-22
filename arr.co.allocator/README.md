# Allocator
## Main Funcs
- Lend some memory.
- Free memory.


## Types of allocation
- Linear Allocation: moving the pointer forward and poping when freeing.
- Using Bitmaps: Using a "bitmap" to keep track of allocated blocks. 
Basically its an "array" where bit represents if that memory block is already allocated or not.
If allocated 1 else 0.
    - init memory, defining total dynamic memory size.
    - Construct bitmap: 1 bit per block.
A block can be of any size: 1 byte, 1 mb, 1gb, etc.

### Granularity vs Space Ocupation has an important part in this topic
- 1 byte: finest Granularity, large space occup.
- larger: coarser Granularity, less space occup.

## malloc
```
malloc(num_bytes)
let b = ceil(num_bytes / bytes_per_block)
```

## Allocation Policies
- First Fit: scan until find any spot large enough to contain data.
- Best fit: scan whole bitmap, return smalles spot large enough to contain data.

## free()
Will set bitmap sets to zero.

## Stack Memory
- Automatically handled by the compiler.
- Very fast alloaction.
- Very fast cleanup.

- Fixed amount of memory.
- Fixed sizes.
- Fixed lifetimes.




# Least Recently Used (LRU) Allocator

Algorithm that tracks memory usage by keeping track of how recently each block of
memory was accessed. The "least recently used" block is the one that hasn't been 
used in the longest time. When the system needs more memory, the LRU allocator 
identifies the least recently used block and frees or reuses it for new allocations.

## How does and LRU allocator work?
1. Tracking usage: It maintains a list or some from of metadata to track when each 
memory block was last used. A typical data structure for this is a linked list or 
a doubly linked list.
2. Allocating Memory:  When a new memory request comes in, the allocator either finds
an unused block or reclaims one from the least recently used blocks.
3. Deallocating Memory: When memory needs to bre freed, the allocator uses the LRU
Algorithm to determine which block has been used the least recently and then
reallocates it.
4. Eviction Policy: If there is no more memory available, the allocator will "evict" the
least recently used block of memory to satisfy the new memory request.

## Use Cases
- Catching systems
- Memory Management
- Garbage Collection

## LRUAllocator
1. Cache Dictionary: 
    - The `LRUAllocator` maintains a cache `self.cache` that stores reusable buffers. The cache
    is a dictionary where the keys are tuples of buffer size and options, and the values are 
    lists of available buffers. These buffers can be reused without needing to reallocate memory.
2. Allocation(alloc): 
    - When allocating a buffer, the allocator first checks if a buffer of the required size and options
    is available in the cache.
    - If a buffer is found `c.pop()` it is reused instead of creating a new one. This saves time and 
    resources.
    - If no cached buffer is available, it tries to allocate a new buffer using the superclass allocator
    (`super().alloc(size, options)`).
    - In case of allocation failure due to memory constraints, the allocator calls `free_cache()` to free up
    cached buffers and retries the allocation.
3. Freeing Buffers(free):
    - When a buffer is no longer needed, it is not inmediately deallocated. Instead, it is placed in
    the cache if LRU caching is enabled (`getenv(LRU, 1)`) checks this environment variable.
    - Only if LRU caching is disabled or the buffer has a specific option(`nolru`), the buffer is 
    inmediately deallocated by calling `super().free()`.
4. Freeing the Cache(`free_cache`):
    - When the system runs out of memory or resources, the allocator can clear the entire cache by calling
    `free_cache()`. This deallocates all cached buffers, effectively resetting the cache.

## Concepts
- dataclasses python: class typically containing mainly data. Attributes require typing. You can set default values.
- dataclasses frozen: protect all fields so that they can be modified only in a way we want. Frozen True automatically
adds `__deleteattr__` and `__setattr__` methods for each field so that they are proected from deletition or updates
after initialization. Others wont be able to add new fields as well.
