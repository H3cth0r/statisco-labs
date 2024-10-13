import ctypes
import mmap
from text import code

DTYPE_INT = 0
DTYPE_FLOAT = 1
DTYPE_DOUBLE = 2
DTYPE_CHAR = 3

class Data(ctypes.Union):
    _fields_ = [
        ("int_data", ctypes.POINTER(ctypes.c_int)),
        ("float_data", ctypes.POINTER(ctypes.c_float)),
        ("double_data", ctypes.POINTER(ctypes.c_double)),
        ("char_data", ctypes.c_char_p),
    ]

class GenericArray(ctypes.Structure):
    _fields_ = [
        ("dtype", ctypes.c_int),
        ("data", Data),
        ("size", ctypes.c_size_t)
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
        # CODE_SIZE = 10000000
        CODE_SIZE = 10000
        code_address = mmap_function(None, 
                                     CODE_SIZE,
                                     mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
                                     mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                                     -1, 0)

        libc.malloc.argtypes = [ctypes.c_size_t]
        libc.malloc.restype = ctypes.c_void_p
        libc.free.argtypes = [ctypes.c_void_p]
        ga_size = 100
        genericArrPtr = libc.malloc(ga_size*ctypes.sizeof(GenericArray))

        dataPtr = libc.malloc(size*ctypes.sizeof(dtype))
        if not dataPtr:
            raise MemoryError("Failed to allocate memory for array data")

        if code_address == -1:
            raise OSError("mmap failed to allocate memory")
        assert len(code) <= CODE_SIZE
        ctypes.memmove(code_address, code, len(code))

        init_array_type = ctypes.CFUNCTYPE(ctypes.POINTER(GenericArray), ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p)
        self._init_array = init_array_type((code_address + 0x1139 - 0x1080))

        free_array_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(GenericArray))
        self._free_array = free_array_type((code_address + 0x11dd - 0x1080))

        set_element_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(GenericArray), ctypes.c_size_t, ctypes.c_void_p)
        self._set_element = set_element_type((code_address + 0x122d - 0x1080))

        get_element_type = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.POINTER(GenericArray), ctypes.c_size_t)
        self._get_element = get_element_type((code_address + 0x128f - 0x1080))

        set_all_elements_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(GenericArray), ctypes.c_void_p)
        self._set_all_elements = set_all_elements_type((code_address + 0x12ed - 0x1080))

        test_method_type = ctypes.CFUNCTYPE(ctypes.c_int)
        self._test_method = test_method_type((code_address + 0x1321 - 0x1080))

        ctype_val = 3
        if dtype == ctypes.c_int:
            ctype_val = 0
        elif dtype == ctypes.c_float:
            ctype_val = 1
        elif dtype == ctypes.c_double:
            ctype_val = 2

        self.array = self._init_array(ctype_val, size, genericArrPtr, dataPtr)
        # self.array = ctypes.cast(genericArrPtr, ctypes.POINTER(GenericArray)) # This can be commented
        if not self.array: raise MemoryError("Failed to initialize array")
        print("Array initialized successfully")

    def set(self, index, value): 
        value_ptr = ctypes.pointer(ctypes.c_int(value))
        self._set_element(self.array, index, value_ptr)
    # def get(self, index): 
    #     res = self._get_element(self.array, index) 
    #     print(dir(res))
    #     return res
    def get(self, index):
        res_ptr = self._get_element(self.array, index)
        if not res_ptr:
            raise IndexError("Index out of bounds")
        
        if self.array.contents.dtype == DTYPE_INT:
            return ctypes.cast(res_ptr, ctypes.POINTER(ctypes.c_int)).contents.value
        elif self.array.contents.dtype == DTYPE_FLOAT:
            return ctypes.cast(res_ptr, ctypes.POINTER(ctypes.c_float)).contents.value
        elif self.array.contents.dtype == DTYPE_DOUBLE:
            return ctypes.cast(res_ptr, ctypes.POINTER(ctypes.c_double)).contents.value
        elif self.array.contents.dtype == DTYPE_CHAR:
            return ctypes.cast(res_ptr, ctypes.POINTER(ctypes.c_char)).contents.value
        else:
            raise ValueError("Unknown data type")

    def __del__(self):
        print("\n\n")
        self._free_array(self.array)
        print("Array freed successfully")
