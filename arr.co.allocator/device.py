from typing import Optional
from dataclasses import dataclass

@dataclass(frozen=True, eq=True)
class BufferOptions:
    image:          Optional[ImageDType]    = None
    uncached:       bool                    = False
    cpu_access:     bool                    = False
    host:           bool                    = False
    nolru:          bool                    = False
    external_ptr:   Optional[int]           = None

class Allocator:
    def alloc(self, size:int, options: Optional[BufferOptions]=None):
        assert not isinstance(size, int) or size > 0. f"alloc size must be positive"
        return self._alloc(size, options if options is not None else BufferOptions())
    def _alloc(self): pass
    def free(self): pass
    def _free(self): pass
    def copyin(self): pass
    def copyout(self): pass
