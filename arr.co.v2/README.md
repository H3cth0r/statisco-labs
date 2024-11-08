

```py
@dataclass(frozen=True, eq=True)
class BufferOptions:
    """
    Configuration options for buffer allocation.
    
    Attributes:
    - image: Optional image data type associated with the buffer.
    - uncached: Boolean flag to determine if caching should be disabled.
    - cpu_access: Boolean flag for enabling CPU access to the buffer.
    - host: Boolean flag indicating if buffer is allocated on the host.
    - nolru: Boolean flag to disable Least Recently Used (LRU) caching.
    - external_ptr: Optional external pointer for the buffer.
    """
    image:          Optional[ImageDType]    = None
    uncached:       bool                    = False
    cpu_access:     bool                    = False
    host:           bool                    = False
    nolru:          bool                    = False
    external_ptr:   Optional[int]           = None
```
```py
class _MallocAllocator(LRUAllocator):
    """
    Allocator for memory using malloc and ctypes

    Methods:
    - _alloc: Allocates memory using ctypes, potentially with an external pointer.
    - as_buffer: Returns a memoryview for the given source buffer.
    - copyin: copies data from a memoryview into the destination buffer.
    - copyout: copies data from the source buffer into a memoryview.
    - offset: returns a slice of the buffer with an applied offset.
    """
    def _alloc(self, size:int, options:BufferOptions):
        """ Allocates memory for size butes, using an external pointer if provided """
        return (ctypes.c_uint8 * size).from_address(options.external_ptr) if options.external_ptr else (ctypes.c_uint8 * size)()
    def as_buffer(self, src) -> memoryview: 
        """ Returns a flattened memoryview of src. """
        return flat_mv(memoryview(src))
    def copyin(self, dest, src:memoryview): 
        """ Copies data from src memoryview into dest buffer. """
        ctypes.memmove(dest, from_mv(src), len(src))
    def copyout(self, dest:memoryview, src): 
        """ Copies data from src buffer into dest memoryview. """
        ctypes.memmove(from_mv(dest), src, len(dest))
    def offset(self, buf, size:int, offset:int): 
        """ Returns a buffer slice starting from offset of size bytes. """
        return from_mv(self.as_buffer(buf)[offset:offset+size])
MallocAllocator = _MallocAllocator()
```

- `_alloc`: allocates the memory given a `size`. If options is not `None`, then
it uses `.from_address()`, which creates a ctypes object at a specific memory
address. It treats the memory at that address as an array of `c_uint8` values.
For instance, the following lines attempts to read 8 bytes starting from
memory address `1`, interpreting them as an array of `c_uint8` values.
```
(ctypes.c_uint8 * 8).from_address(1)
```
This is unsafae and can lead to segmentation faults or other errors, depending
on your  operating system and Python interpreter. If there is specific
data you want to read from a known memory address, like within mapped 
memory space or hardware buffer), using `from_address` with the correct and
valid address is appropiate.
```
>>> external_memory_address = 0x12345678  
>>> (ctypes.c_uint8 * 8).from_address(external_memory_address)
<__main__.c_ubyte_Array_8 object at 0x7fd4c58925c0>
```
- `as_buffer`: returns flattened memoryview of src. The function takes a `memoryview`
object as input. A `memoryview` provides a way to access the internal data of an
object without copying it, allowing  to work with the data more efficiently. 
This `mv.cast("B", shape=(mv.nbytes, ))` is flattening. `.cast("B")` reinterprets
the data as a contiguous array of bytes (B stands form `c_ubyte`). Then `shape=(mv.nbytes, )`
reshapes the memoryview to have a single dimension with a length equal to the total
number of bytes in `mv`.
