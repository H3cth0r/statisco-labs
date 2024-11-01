from __future__ import annotations
from collections import defaultdict
from typing import Optional, Dict, Tuple, Any, Union, Final
import ctypes, functools
from dataclasses import dataclass

ConstType = Union[float, int, bool]

@dataclass(frozen=True)
class DType:
    priority: int
    itemsize: int
    name: str
    fmt: Optional[str]
    count: int
    def __repr__(self): return f"dtypes.{INVERSE_DTYPES_DICT[self.scalar().name]}"+(f".vec({self.count})" if self.count > 1 else "")
    def __lt__(self, o:DType): return (self.priority, self.itemsize, self.name, self.fmt, self.count) < (o.priority, o.itemsize, o.name, o.fmt, o.count)
    @property
    def base(self): return self
    @property
    def vcount(self): return self.count
    def vec(self, sz:int) -> DType:
        assert self.count == 1, f"can't vectorize {self} with size {sz}"
        if sz == 1 or self.name == "void": return self
        return DType(self.priority, self.itemsize*sz, f"{INVERSE_DTYPES_DICT[self.name]}{sz}", None, sz)
    def ptr(self, local=False) -> Union[PtrDType, ImageDType]:
        return PtrDType(self.priority, self.itemsize, self.name, self.fmt, self.count, self, local)
    def scalar(self) -> DType: return DTYPES_DICT[self.name[:-len(str(self.count))]] if self.count > 1 else self

@dataclass(frozen=True)
class PtrDType(DType):
    _base: DType
    local: bool = False
    v: int = 1
    @property
    def base(self): return self._base
    def scalar(self) -> PtrDType: return replace(self, v=1)
    def vec(self, sz: int) -> PtrDType: return replace(self, v=sz)
    def ptr(self, local=False): raise RuntimeError("can't make a pointer from a pointer")
    @property
    def vcount(self): return self.v
    def __repr__(self): return f"{self.base.__repr__()}.ptr({'local=true' if self.local else ''})" + (f".vec({self.v})" if self.v != 1 else "")
@dataclass(frozen=True)
class ImageDType(PtrDType):
    shape: Tuple[int, ...] = ()
    def ptr(self, local=False) -> Union[PtrDType, ImageDType]:
        assert not local, "images can't be local"
        return self
    def __repr__(self): return f"dtypes.{self.name}({self.shape})" + (f".vec({self.v})" if self.v != 1 else "")

class dtypes:
    @staticmethod
    @functools.lru_cache(None)
    def if_float(x: DType) -> bool: return x.scalar() in dtypes.floats or isinstance(x, ImageDType)
    @staticmethod
    @functools.lru_cache(None)
    def is_int(x: DType) -> bool: return x.scalar() in dtypes.ints
    @staticmethod
    @functools.lru_cache(None)
    def is_unsigned(x: DType) -> bool: return x.scalar() in dtypes.uints
    @staticmethod
    def from_py(x) -> DType:
        if x.__class__ is float: return dtypes.default_float
        if x.__class__ is int: return dtypes.default_int
        if x.__class__ is bool: return dtypes.bool
        if x.__class__ is list or x.__class__ is tuple: return max(dtypes.from_py(xi) for xi in x) if x else dtypes.default_float
        raise RuntimeError(f"Could not infer dtype of {x} with type {type(x)}")
    @staticmethod
    def as_const(val: Tuple[ConstType, ...] |ConstType, dtype:DType):
        if isinstance(val, tuple):
            assert len(val) == dtype.count, f"mismatch {val} {dtype}"
            return tuple(dtypes.as_const(x, dtype) for x in val)
        return int(val) if dtypes.is_int(dtype) else float(val) if dtypes.is_float(dtype) else bool(val)
    @staticmethod
    @functools.lru_cache(None)
    def min(dtype:DType):
        if dtypes.is_int(dtype): return 0 if dtypes.is_unsigned(dtype) else -2**(dtype.itemsize*8-1)
        return -float("inf") if dtypes.is_float(dtype) else False
    @staticmethod
    @functools.lru_cache(None)
    def max(dtype:DType):
        if dtypes.is_int(dtype): return (2**(dtype.itemsize*8-(0 if dtypes.is_unsigned(dtype) else 1)))-1
        return float("inf") if dtypes.is_float(dtype) else True
    @staticmethod
    def finfo(dtype:DType) -> Tuple[int, int]:
        if not dtypes.is_float(dtype): raise ValueError(f"{dtype} is not a floating point type")
        return {dtypes.float16: (5, 10), dtypes.bfloat16: (8, 7), dtypes.float32: (8, 23), dtypes.float64: (11, 52)}[dtype]
    @staticmethod
    def fields() -> Dict[str, DType]: return DTYPES_DICT
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

DTYPES_DICT = {k: v for k, v in dtypes.__dict__.items() if not (k.startswith(('__', 'default', 'void'))
                                                                or v.__class__ is staticmethod or isinstance(v, tuple))}
INVERSE_DTYPES_DICT = {v.name: k for k, v in DTYPES_DICT.items()}
INVERSE_DTYPES_DICT['void'] = 'void'



@dataclass(frozen=True, eq=True)
class BufferOptions:
    image:          Optional[ImageDType]    = None
    uncached:       bool                    = False
    cpu_access:     bool                    = False
    host:           bool                    = False
    nolru:          bool                    = False
    external_ptr:   Optional[int]           = None


class Allocator:
    def alloc(self, size: int, options: Optional[BufferOptions]=None):
        assert not isinstance(size, int) or size > 0, f"alloc size must be positive, getting {size}"
        return self._alloc(size, options if options is not None else BufferOptions())
    def _alloc(self, size: int, options: BufferOptions): raise NotImplementedError("need alloc")
    def free(self, opaque, size: int, options:Optional[BufferOptions]=None): self._free(opaque, options if options is not None else BufferOptions())
    def _free(self, opaque, options: BufferOptions): pass
    def copyin(self, dest, src: memoryview): raise NotImplementedError("need copyin")
    def copyout(self, dest: memoryview, src): raise NotImplementedError("need copyout")

class LRUAllocator(Allocator):
    """
    responsible for caching buffers. Ensures buffers are not freed until 
    absolutely necessary. Optimizes performance.
    """
    def __init__(self): self.cache: Dict[Tuple[int, Optional[BufferOptions]], Any] = defaultdict(list)
    def alloc(self, size: int, options: Optional[BufferOptions]=None):
        if len(c := self.cache[(size, options)]): return c.pop()
        try: return super().alloc(size, options)
        except (RuntimeError, MemoryError):
            self.free_cache()
            return super().alloc(size, options)
    def free_cache(self):
        for (sz, options), opaques in self.cache.items():
            for opaque in opaques: super().free(opaque, sz, options)
            opaques.clear()
    def free(self, opaque: Any, size: int, options: Optional[BufferOptions]=None):
        if getenv("LRU", 1) and (options is None or not options.nolru): self.cache[(size, options)].append(opaque)
        else: super().free(opaque, size, options)

def flat_mv(mv:memoryview): return mv if len(mv) == 0 else mv.cast("B", shape=(mv.nbytes, ))
def from_mv(mv:memoryview, to_type=ctypes.c_char):
    return ctypes.cast(ctypes.addressof(to_type.from_buffer(mv)), ctypes.POINTER(to_type * len(mv))).contents

class _MallocAllocator(LRUAllocator):
    def _alloc(self, size:int, options:BufferOptions):
        return (ctypes.c_uint8 * size).from_address(options.external_ptr) if options.external_ptr else (ctypes.c_uint8 * size)()
    def as_buffer(self, src) -> memoryview: return flat_mv(memoryview(src))
    def copyin(self, dest, src:memoryview): ctypes.memmove(dest, from_mv(src), len(src))
    def copyout(self, dest:memoryview, src): ctypes.memmove(from_mv(dest), src, len(dest))
    def offset(self, buf, size:int, offset:int): return from_mv(self.as_buffer(buf)[offset:offset+size])
MallocAllocator = _MallocAllocator()
