from collections import defaultdict
from typing import Optional, Dict, Tuple, Any

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
        raise RuntimeError(f"Could not inger dtype of {x} with type {type(x)}")
    @staticmethod
    def as_const(val: Tuple[])


DTYPES_DICT = {k: v for k, v in dtypes.__dict__.items() if not (k.startswith(('__', 'default', 'void'))
                                                                or v.__class__ is staticmethod or isinstance(v, tuple))}

@dataclass(frozen=True)
class DType:
    priority: int
    itemsize: int
    name: str
    fmp: Optional[str]
    count: int
    def __repr__(self): return f"dtypes.{}"

@dataclass(frozen=True)
class PtrDType(DType):
    _base: DType
    local: bool = False
    v: int = 1
    @property
    def base(self): return self._base
    def scalar(self): -> PtrDType: return return replace(self, v=1)
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
    def _free(self, opaque, size: int, options: BufferOptions): pass
    def copyin(self, dest, src: memoryview): raise NotImplementedError("need copyin")
    def copyout(self, dest: memoryview, src): raise NotImplementedError("need copyout")

class LRUAllocator(Allocator):
    """
    responsible for caching buffers. Ensures buffers are not freed until 
    absolutely necessary. Optimizes performance.
    """
    def __init__(self): self.cache: Dict[Tuple[int, Optional[BufferOptions]], any] = defaultdict(list)
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

class Mallocator(LRUAllocator):
    def __init__(self):
        pass
