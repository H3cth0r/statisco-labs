from typing import Optional, Tuple, Union
from dataclasses import dataclass
import functools

ConstType = Union[float, int, bool]

class dtypes:
    @staticmethod
    @functools.lru_cache(None)
    def is_float(x: Dtype) -> bool: return x.scalar() in dtypes.floats
    @staticmethod
    @functools.lru_cache(None)
    def is_int(x: Dtype) -> bool: return x.scalar() in dtypes.ints
    @staticmethod
    @functools.lru_cache(None)
    def is_unsigned(x: Dtype) -> bool: return x.scalar() in dtypes.uints
    @staticmethod
    def from_py(x) -> Dtype:
        if x.__class__ is float: return dtypes.default_float
        if x.__class__ is int: return dtypes.default_int
        if x.__class__ is bool: return dtypes.bool
        if x.__class__ is list or x.__class__ is tuple: return max(dtypes.from_py(xi) for xi in x) if x else dtypes.default_float
        raise RuntimeError(f"Could not infer dtype of {x} with type {type(x)}")
    @staticmethod
    def as_const(val: Tuple[ConstType, ...]|ConstType, dtypes: Dtype):
        if isinstance(val, tuple):
            assert len(val) == dtype.count, f"mismatch {val} {dtype}"
            return tuple(dtypes.as_const(x, dtype) for x in val)
        return int(val) if dtypes.is_int(dtype) else float(val) if dtypes.is_float(dtype) else bool(val)
    @staticmethod
    @functools.lru_cache(None)
    def min(dtype: Dtype):
        if dtypes.is_int(dtype): return 0 if dtypes.is_unsigned(dtype) else -2**(dtype.itemsize*8-1)

DTYPES_DICT = {k: v for k, v in dtypes.__dict__.items() if not (k.startswith(('__', 'default', 'void')) 
                                                                or v.__class__ is staticmethod or isinstance(v, tuple))}
INVERSE_DTYPES_DICT = 

class Dtype:
    priority:   int
    itemsize:   int
    fmt:        Optional[str]
    count:      int
    def __repr__(self): return f"dtypes."
    def vec(self, sz: int):
