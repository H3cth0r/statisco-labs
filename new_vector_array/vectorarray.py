import ctypes
import mmap
from text import code

class data(ctypes.Union):
    _fields_ = [
            ("int_data",        ctypes.POINTER(ctypes.c_int)),
            ("float_data",      ctypes.POINTER(ctypes.c_float)),
            ("double_data",     ctypes.POINTER(ctypes.c_double)),
            ("char_data",       ctypes.c_char_p),
    ]
class GenericArray(ctypes.Structure):
    _fields_ = [
            ("dtype",   ctypes.c_int),
            ("data",    data),
            ("size",    ctypes.c_size_t)
    ]

class VectorArray:
    def __init__(self, size, dtype=ctypes.c_int):
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

        _init_array_type        = ctypes.CFUNCTYPE(ctypes.POINTER(GenericArray), ctypes.c_int, ctypes.c_size_t)
        self._init_array        = ctypes.cast(code_address+0x1139 - 0x1080, _init_array_type)

        _free_array_type        = ctypes.CFUNCTYPE(None, ctypes.POINTER(GenericArray))
        self._free_array        = ctypes.cast(code_address+0x11f0 - 0x1080, _free_array_type)
        
        _set_element_type       = ctypes.CFUNCTYPE(None, ctypes.POINTER(GenericArray), ctypes.c_size_t, ctypes.c_void_p)
        self._set_element       = ctypes.cast(code_address+0x125a - 0x1080, _set_element_type)

        _get_element_type       = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.POINTER(GenericArray), ctypes.c_size_t)
        self._get_element       = ctypes.cast(code_address+0x12bc - 0x1080, _get_element_type)

        _set_all_elements_type  = ctypes.CFUNCTYPE(None, ctypes.POINTER(GenericArray), ctypes.c_void_p)
        _set_all_elements       = ctypes.cast(code_address+0x131a - 0x1080, _set_all_elements_type)

        _test_method_type       = ctypes.CFUNCTYPE(ctypes.c_int)
        _test_method            = ctypes.cast(code_address+0x1334 - 0x1080, _test_method_type)

        test_res = _test_method()
        print(test_res)

        ctype_val = 3
        if dtype == ctypes.c_int:
            ctype_val = 0
        elif dtype == ctypes.c_float:
            ctype_val = 1
        elif dtype == ctypes.c_double:
            ctype_val = 2

        self.array              = self._init_array(ctype_val, size)
        print("Here6")

    def __del__(self):
        self._free_array(self.array)
