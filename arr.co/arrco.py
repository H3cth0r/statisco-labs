import ctypes
import mmap
from byteArrCo import code, text_start, init_array_address, set_element_address, get_element_address, set_all_elements_address 

DTYPE_INT = 0
DTYPE_FLOAT = 1
DTYPE_DOUBLE = 2
DTYPE_CHAR = 3

class Data(ctypes.Union):
    _fields_ = [
        ("int_data",    ctypes.POINTER(ctypes.c_int)),
        ("float_data",  ctypes.POINTER(ctypes.c_float)),
        ("double_data", ctypes.POINTER(ctypes.c_double)),
        ("char_data",   ctypes.c_char_p),
    ]

class ArrCoStruct(ctypes.Structure):
    _fields_ = [
        ("dtype",   ctypes.c_int),
        ("data",    Data),
        ("size",    ctypes.c_size_t)
    ]

class array:
    max_print = 100
    def __init__(self, size, dtype=ctypes.c_double):
        self.libc               = ctypes.cdll.LoadLibrary(None)
        mmap_function           = self.libc.mmap
        mmap_function.restype   = ctypes.c_void_p
        mmap_function.argtypes  = (
            ctypes.c_void_p,    ctypes.c_size_t,
            ctypes.c_int,       ctypes.c_int,
            ctypes.c_int,       ctypes.c_size_t
        )
        CODE_SIZE               = 10000
        code_address            = mmap_function(None,
                                                CODE_SIZE,
                                                mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
                                                mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                                                -1, 0)

        self.libc.malloc.argtypes   = [ctypes.c_size_t]
        self.libc.malloc.restype    = ctypes.c_void_p
        self.libc.free.argtypes     = [ctypes.c_void_p]

        arrco_size          = 10
        self.arrcoPtr       = self.libc.malloc(arrco_size * ctypes.sizeof(ArrCoStruct))
        if not self.arrcoPtr: raise MemoryError("Failed to allocate memory for arrco pointer")
        
        self.dataPtr        = self.libc.malloc(size*ctypes.sizeof(dtype))
        if not self.dataPtr: raise MemoryError("Failed to allocate memory for data pointer")

        if code_address == -1: raise OSError("mmap failed to allocate memory")
        assert len(code) <= CODE_SIZE
        ctypes.memmove(code_address, code, len(code))
        
        init_array_type         = ctypes.CFUNCTYPE(ctypes.POINTER(ArrCoStruct), ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p)
        self._init_array        = init_array_type((code_address + init_array_address - text_start))

        set_element_type        = ctypes.CFUNCTYPE(None, ctypes.POINTER(ArrCoStruct), ctypes.c_size_t, ctypes.c_void_p)
        self._set_element       = set_element_type((code_address + set_element_address - text_start))

        get_element_type        = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.POINTER(ArrCoStruct), ctypes.c_size_t)
        self._get_element       = get_element_type((code_address + get_element_address - text_start))

        set_all_elements_type   = ctypes.CFUNCTYPE(None, ctypes.POINTER(ArrCoStruct), ctypes.c_void_p)
        self._set_all_elements  = set_all_elements_type((code_address + set_all_elements_address - text_start))

        ctype_val = 3
        if dtype == ctypes.c_int: ctype_val = 0
        elif dtype == ctypes.c_float: ctype_val = 1
        elif dtype == ctypes.c_double: ctype_val = 2

        self.array  = self._init_array(ctype_val, size, self.arrcoPtr, self.dataPtr)
        if not self.array: raise MemoryError("Failed to initialize array")
    def set(self, index, value):
        value_ptr   = ctypes.pointer(ctypes.c_int(value))
        self._set_element(self.array, index, value_ptr)
    def get(self, index):
        res_ptr = self._get_element(self.array, index)
        if not res_ptr: raise IndexError("Index out of bounds")
        dtype = self.array.contents.dtype
        if dtype == DTYPE_INT: return ctypes.cast(res_ptr, ctypes.POINTER(ctypes.c_int)).contents.value
        elif dtype == DTYPE_FLOAT: return ctypes.cast(res_ptr, ctypes.POINTER(ctypes.c_float)).contents.value
        elif dtype == DTYPE_DOUBLE: return ctypes.cast(res_ptr, ctypes.POINTER(ctypes.c_double)).contents.value
        elif dtype == DTYPE_CHAR: return ctypes.cast(res_ptr, ctypes.POINTER(ctypes.c_char)).contents.value
        else: raise ValueError("Unknown data type")
    def ones(self): 
        value_ptr   = ctypes.pointer(ctypes.c_int(1))
        self._set_all_elements(self.array, value_ptr)
    def __del__(self):
        if hasattr(self, 'libc'):
            if hasattr(self, 'arrcoPtr'): self.libc.free(self.arrcoPtr)
            if hasattr(self, 'dataPtr'): self.libc.free(self.dataPtr)
    def tolist(self):
        dtype = self.array.contents.dtype
        if dtype == DTYPE_INT: return self.array.contents.data.int_data[:self.array.contents.size]
        elif dtype == DTYPE_FLOAT: return self.array.contents.data.float_data[:self.array.contents.size]
        elif dtype == DTYPE_DOUBLE: return self.array.contents.data.double_data[:self.array.contents.size]
        elif dtype == DTYPE_CHAR: return self.array.contents.data.char_data[:self.array.contents.size]
        else: raise ValueError("Unknown data type")
    def __str__(self):
        dtype, size = self.array.contents.dtype, self.array.contents.size
        if dtype == DTYPE_INT: type_str = 'int32'
        elif dtype == DTYPE_FLOAT: type_str = 'float32'
        elif dtype == DTYPE_DOUBLE: type_str = 'float64'
        elif dtype == DTYPE_CHAR: type_str = 'char'
        else: type_str = 'Unknown'
        if size <= self.max_print:
            data        = self.tolist()
            data_str    = ', '.join(map(str, data))
        else:
            start       = self.max_print // 8
            end         = size - start
            if dtype == DTYPE_INT:
                left    = self.array.contents.data.int_data[:start]
                right   = self.array.contents.data.int_data[end:size]
            elif dtype == DTYPE_FLOAT: 
                left    = self.array.contents.data.float_data[:start]
                right   = self.array.contents.data.float_data[end:size]
            elif dtype == DTYPE_DOUBLE: 
                left    = self.array.contents.data.double_data[:start]
                right   = self.array.contents.data.double_data[end:size]
            elif dtype == DTYPE_CHAR:
                left    = self.array.contents.data.char_data[:start]
                right   = self.array.contents.data.char_data[end:size]
            data_str = ', '.join(map(str, left)) + ', ...,' + ', '.join(map(str, right))
        return f"array([{data_str}], dtype={type_str})"
    def __repr__(self):return self.__str__()
