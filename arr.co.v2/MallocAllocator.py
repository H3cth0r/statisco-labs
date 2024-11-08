import ctypes
from typing import Optional, Any
import array

# Pre-compile frequently used C types
_c_uint8_array = ctypes.c_uint8 * 1
_memmove = ctypes.memmove
_addressof = ctypes.addressof

# Pre-define common buffer sizes and keep them ready
_BUFFER_CACHE = {}
def _init_buffer_cache():
    global _BUFFER_CACHE
    common_sizes = [4, 8, 16, 32, 64, 128, 256]  # Add sizes you commonly use
    for size in common_sizes:
        _BUFFER_CACHE[size] = (ctypes.c_uint8 * size)

# Initialize cache at module import
_init_buffer_cache()

def alloc(size: int, external_ptr: Optional[int] = None) -> Any:
    """Optimized memory allocation."""
    if external_ptr:
        return (ctypes.c_uint8 * size).from_address(external_ptr)
    
    # Use cached buffer type if available
    buffer_type = _BUFFER_CACHE.get(size)
    if buffer_type is None:
        buffer_type = ctypes.c_uint8 * size
        # Cache the new buffer type if it's likely to be reused
        if size <= 1024:  # Only cache reasonable sizes
            _BUFFER_CACHE[size] = buffer_type
    
    return buffer_type()

def copyin(dest: Any, src: memoryview) -> None:
    """Optimized memory copy."""
    _memmove(dest, src.tobytes(), len(src))

def copyout(dest: memoryview, src: Any) -> None:
    """Optimized memory copy out."""
    _memmove(dest.tobytes(), src, len(dest))

def free(buffer: Any) -> None:
    """Quick no-op for compatibility."""
    pass

# Optional: Pre-allocate common small buffers
_PREALLOCATED = {
    4: alloc(4),
    8: alloc(8),
    16: alloc(16)
}

def get_preallocated(size: int) -> Optional[Any]:
    """Get a pre-allocated buffer if available."""
    return _PREALLOCATED.get(size)
