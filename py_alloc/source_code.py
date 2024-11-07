from __future__ import annotations
from collections import defaultdict
from typing import Optional, Dict, Tuple, Any, Union, Final
import ctypes, functools
from dataclasses import dataclass
from helpers import getenv

ConstType = Union[float, int, bool]

@dataclass(frozen=True)
class DType:
    """
    Base data type class to define the properties of various data types.
    Attributes:
        priority (int): The priority level of the data type.
        itemsize (int): The size of an individual element in bytes.
        name (str): The name of the data type.
        fmt (Optional[str]): The format string for the data type, if applicable.
        count (int): The count of elements (used for vectorized types).
    """
    priority: int
    itemsize: int
    name: str
    fmt: Optional[str]
    count: int

    def __repr__(self): 
        """ Provides a readable string representation of the data type. """
        return f"dtypes.{INVERSE_DTYPES_DICT[self.scalar().name]}"+(f".vec({self.count})" if self.count > 1 else "")
    def __lt__(self, o:DType): 
        """ Defines a comparion for sorting based on type priority, size, name, format and count. """
        return (self.priority, self.itemsize, self.name, self.fmt, self.count) < (o.priority, o.itemsize, o.name, o.fmt, o.count)
    @property
    def base(self): 
        """ Returns the base type of the data type. """
        return self
    @property
    def vcount(self): 
        """ Returns the count of vectorized elements. """
        return self.count
    def vec(self, sz:int) -> DType:
        """ Creates a vectorized version of this data type with a specified size. """
        assert self.count == 1, f"can't vectorize {self} with size {sz}"
        if sz == 1 or self.name == "void": return self
        return DType(self.priority, self.itemsize*sz, f"{INVERSE_DTYPES_DICT[self.name]}{sz}", None, sz)
    def ptr(self, local=False) -> Union[PtrDType, ImageDType]:
        """ Returns a pointer type for this data type. """
        return PtrDType(self.priority, self.itemsize, self.name, self.fmt, self.count, self, local)
    def scalar(self) -> DType: 
        """ Returns the scalar (non-vectorized) version of this data type """
        return DTYPES_DICT[self.name[:-len(str(self.count))]] if self.count > 1 else self

@dataclass(frozen=True)
class PtrDType(DType):
    """
    A specialized data type class for pointer types, inheriting from Dtype.

    Attributes:
        _base (Dtype): The base data type this pointer points to.
        local (book): Specifies if the pointer is a local pointer.
        v (int): Vectorized count for the pointer type.
    """
    _base: DType
    local: bool = False
    v: int = 1
    @property
    def base(self): 
        """ Returns the base data type of the pointer. """
        return self._base
    def scalar(self) -> PtrDType: 
        """ Returns the scalar (non-vectorized) version of this pointer type. """
        return replace(self, v=1)
    def vec(self, sz: int) -> PtrDType: 
        """ Returns a vectorized version of this pointer type with the specified size. """
        return replace(self, v=sz)
    def ptr(self, local=False): 
        """ Raises an Error, as pointer cannot be further converted into pointer types. """
        raise RuntimeError("can't make a pointer from a pointer")
    @property
    def vcount(self): 
        """ Returns the vectorized count of this pointer type. """
        return self.v
    def __repr__(self): 
        """ Provides a readable string representation of the pointer type. """
        return f"{self.base.__repr__()}.ptr({'local=true' if self.local else ''})" + (f".vec({self.v})" if self.v != 1 else "")
@dataclass(frozen=True)
class ImageDType(PtrDType):
    """
    A specialized data type class for image types, inheriting form PtrDType.

    Attributes:
        shape (Tuple[int, ...]): the shape of the image type.
    """
    shape: Tuple[int, ...] = ()
    def ptr(self, local=False) -> Union[PtrDType, ImageDType]:
        """ Returns the pointer to the image, ensuring it is not local. """
        assert not local, "images can't be local"
        return self
    def __repr__(self): 
        """ Provides a readable string representation of the image type. """
        return f"dtypes.{self.name}({self.shape})" + (f".vec({self.v})" if self.v != 1 else "")

class dtypes:
    """
    Utility class providing various static methods and constants to define, check
    and manage data types.
    """
    @staticmethod
    @functools.lru_cache(None)
    def if_float(x: DType) -> bool: 
        """ Checks if the data type is a floating point or an image data type. """
        return x.scalar() in dtypes.floats or isinstance(x, ImageDType)
    @staticmethod
    @functools.lru_cache(None)
    def is_int(x: DType) -> bool: 
        """ Checks if the data type is an integer type. """
        return x.scalar() in dtypes.ints
    @staticmethod
    @functools.lru_cache(None)
    def is_unsigned(x: DType) -> bool: 
        """ Checks if the data type is an unsigned integer. """
        return x.scalar() in dtypes.uints
    @staticmethod
    def from_py(x) -> DType:
        """ Infers the data type from a Python variable. """
        if x.__class__ is float: return dtypes.default_float
        if x.__class__ is int: return dtypes.default_int
        if x.__class__ is bool: return dtypes.bool
        if x.__class__ is list or x.__class__ is tuple: return max(dtypes.from_py(xi) for xi in x) if x else dtypes.default_float
        raise RuntimeError(f"Could not infer dtype of {x} with type {type(x)}")
    @staticmethod
    def as_const(val: Tuple[ConstType, ...] |ConstType, dtype:DType):
        """ Converts a value constant based on the provided data type. """
        if isinstance(val, tuple):
            assert len(val) == dtype.count, f"mismatch {val} {dtype}"
            return tuple(dtypes.as_const(x, dtype) for x in val)
        return int(val) if dtypes.is_int(dtype) else float(val) if dtypes.is_float(dtype) else bool(val)
    @staticmethod
    @functools.lru_cache(None)
    def min(dtype:DType):
        """ Returns the minimum possible value for a given data type. """
        if dtypes.is_int(dtype): return 0 if dtypes.is_unsigned(dtype) else -2**(dtype.itemsize*8-1)
        return -float("inf") if dtypes.is_float(dtype) else False
    @staticmethod
    @functools.lru_cache(None)
    def max(dtype:DType):
        """ Returns the maximum possible value for a given data type. """
        if dtypes.is_int(dtype): return (2**(dtype.itemsize*8-(0 if dtypes.is_unsigned(dtype) else 1)))-1
        return float("inf") if dtypes.is_float(dtype) else True
    @staticmethod
    def finfo(dtype:DType) -> Tuple[int, int]:
        """ Returns floating-point information (precision and exponent bits) for the data type. """
        if not dtypes.is_float(dtype): raise ValueError(f"{dtype} is not a floating point type")
        return {dtypes.float16: (5, 10), dtypes.bfloat16: (8, 7), dtypes.float32: (8, 23), dtypes.float64: (11, 52)}[dtype]
    @staticmethod
    def fields() -> Dict[str, DType]: 
        """ Returns a dictionary of all registered data types. """
        return DTYPES_DICT
    void: Final[DType] = DType(-1, 0, "void", None, 1)
    bool: Final[DType] = DType(0, 1, "bool", '?', 1)
    int8: Final[DType] = DType(1, 1, "char", 'b', 1)
    uint8: Final[DType] = DType(2, 1, "unsigned char", 'B', 1)
    int16: Final[DType] = DType(3, 2, "short", 'h', 1)
    uint16: Final[DType] = DType(4, 2, "unsigned short", 'H', 1)
    int32: Final[DType] = DType(5, 4, "int", 'i', 1)
    uint32: Final[DType] = DType(6, 4, "unsigned int", 'I', 1)
    int64: Final[DType] = DType(7, 8, "long", 'q', 1)
    uint64: Final[DType] = DType(8, 8, "unsigned long", 'Q', 1)
    float16: Final[DType] = DType(9, 2, "half", 'e', 1)
    bfloat16: Final[DType] = DType(10, 2, "__bf16", None, 1)
    float32: Final[DType] = DType(11, 4, "float", 'f', 1)
    float64: Final[DType] = DType(12, 8, "double", 'd', 1)

    # dtype aliases
    half = float16; float = float32; double = float64 # noqa: E702
    uchar = uint8; ushort = uint16; uint = uint32; ulong = uint64 # noqa: E702
    char = int8; short = int16; int = int32; long = int64 # noqa: E702
    @staticmethod
    def imageh(shp): return ImageDType(100, 2, "imageh", 'e', 1, shape=shp, _base=dtypes.float32)
    @staticmethod
    def imagef(shp): return ImageDType(100, 4, "imagef", 'f', 1, shape=shp, _base=dtypes.float32)

    default_float: ClassVar[DType] = float32
    default_int: ClassVar[DType] = int32

    floats = (float16, bfloat16, float32, float64)
    uints = (uint8, uint16, uint32, uint64)
    sints = (int8, int16, int32, int64)
    ints = uints + sints

# Dictionary of all defined data types mapped to their instances
DTYPES_DICT = {k: v for k, v in dtypes.__dict__.items() if not (k.startswith(('__', 'default', 'void'))
                                                                or v.__class__ is staticmethod or isinstance(v, tuple))}

# Dictionary to map type names to their corresponding type definitions
INVERSE_DTYPES_DICT = {v.name: k for k, v in DTYPES_DICT.items()}
INVERSE_DTYPES_DICT['void'] = 'void'



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


class Allocator:
    """
    Base class for memory allocation operations.

    Methods:
    - alloc: Allocates a buffer of a given size with specified options.
    - free: Frees a previously allocated buffer.
    - copyin: Copies data from source to destination.
    - copyout: Copies data from destination to source.
    """
    def alloc(self, size: int, options: Optional[BufferOptions]=None):
        """ Allocates a buffer of size bytes with options """
        assert not isinstance(size, int) or size > 0, f"alloc size must be positive, getting {size}"
        return self._alloc(size, options if options is not None else BufferOptions())
    def _alloc(self, size: int, options: BufferOptions): 
        """ Internal method for allocating memory. Raises Error if not overriden """
        raise NotImplementedError("need alloc")
    def free(self, opaque, size: int, options:Optional[BufferOptions]=None): 
        """ Frees the buffer identified by opaque and size """
        self._free(opaque, options if options is not None else BufferOptions())
    def _free(self, opaque, options: BufferOptions): 
        """ Internal method for freeing memory """
        pass
    def copyin(self, dest, src: memoryview): 
        """ Copies data into the buffer. Raises Error if not overriden """
        raise NotImplementedError("need copyin")
    def copyout(self, dest: memoryview, src): 
        """ Copies data out of the buffer. Raises Error if not overriden """
        raise NotImplementedError("need copyout")

class LRUAllocator(Allocator):
    """
    responsible for caching buffers. Ensures buffers are not freed until 
    absolutely necessary. Optimizes performance.
    Least Recently Used (LRU) cache

    Methods:
        - alloc: Allocates memory reusing buffer from the cache when possible
        - free_cache: frees all buffers in the cache.
        - free: frees a buffer and adds it to the cache if caching is enabled.
    """
    def __init__(self): 
        """ Initializes and LRU cache for buffer allocation """
        self.cache: Dict[Tuple[int, Optional[BufferOptions]], Any] = defaultdict(list)
    def alloc(self, size: int, options: Optional[BufferOptions]=None):
        """ Allocates a buffer with size bytes, retrieving from cache if available. """
        if len(c := self.cache[(size, options)]): return c.pop()
        try: return super().alloc(size, options)
        except (RuntimeError, MemoryError):
            self.free_cache()
            return super().alloc(size, options)
    def free_cache(self):
        """ Frees all cached buffers to reclaim memory. """
        for (sz, options), opaques in self.cache.items():
            for opaque in opaques: super().free(opaque, sz, options)
            opaques.clear()
    def free(self, opaque: Any, size: int, options: Optional[BufferOptions]=None):
        """ Frees the buffer, caching it if LRU is enabled and options.nolru is False. """
        if getenv("LRU", 1) and (options is None or not options.nolru): self.cache[(size, options)].append(opaque)
        else: super().free(opaque, size, options)

"""
Converts the memoryview to a flat view of bytes

Parameters:
    - mv: Input memoryview object.

Returns:
    - A flattened memoryview in B format
"""
def flat_mv(mv:memoryview): return mv if len(mv) == 0 else mv.cast("B", shape=(mv.nbytes, ))

"""
Converts a memoryview to a ctypes array of a specified type.

Parameters:
    - mv: Input memoryview object.
    - to_type: The ctypes type to convert each element.
Returns:
    - A ctypes array with the converted data.
"""
def from_mv(mv:memoryview, to_type=ctypes.c_char):
    return ctypes.cast(ctypes.addressof(to_type.from_buffer(mv)), ctypes.POINTER(to_type * len(mv))).contents

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
