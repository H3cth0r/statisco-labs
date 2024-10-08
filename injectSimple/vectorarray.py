import ctypes
import mmap
from text import code 

libc = ctypes.cdll.LoadLibrary(None)
mmap_function = libc.mmap
mmap_function.restype = ctypes.c_void_p
mmap_function.argtypes = (
    ctypes.c_void_p, ctypes.c_size_t,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_size_t
)

CODE_SIZE = 10000000
code_address = mmap_function(None, CODE_SIZE,
                             mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
                             mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                             -1, 0)

if code_address == -1: raise OsError("mmap failed to allocate memory")

assert len(code) <= CODE_SIZE
ctypes.memmove(code_address, code, len(code))

additionC_type =  ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int)
additionC = ctypes.cast(code_address+0x00, additionC_type)

substractionC_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int)
substractionC = ctypes.cast(code_address+0x10, substractionC_type)

res = additionC(3, 4)
print(res)

res = substractionC(5, 2)
print(res)

res = additionC(4, 4)
print(res)
