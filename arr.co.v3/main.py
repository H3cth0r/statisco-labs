import ctypes
from ctypes import c_size_t, c_void_p, c_int, c_longlong, Structure, POINTER
import time

class FastArray(Structure):
    _fields_ = [
        ("size", c_size_t),
        ("data", c_void_p),
        ("itemsize", c_size_t)
    ]

class FastArrayInterface:
    def __init__(self, lib_path="./libfastarrayopt.so"):
        self.lib = ctypes.CDLL(lib_path)
        
        # Set up function signatures
        self.lib.create_array.argtypes = [c_size_t, c_size_t]
        self.lib.create_array.restype = POINTER(FastArray)
        
        self.lib.destroy_array.argtypes = [POINTER(FastArray)]
        self.lib.destroy_array.restype = None
        
        self.lib.fill_zeros_optimized.argtypes = [POINTER(FastArray)]
        self.lib.fill_zeros_optimized.restype = None
        
        self.lib.fill_range_optimized.argtypes = [POINTER(FastArray)]
        self.lib.fill_range_optimized.restype = None
        
        self.lib.sum_array_optimized.argtypes = [POINTER(FastArray)]
        self.lib.sum_array_optimized.restype = c_longlong

    def zeros(self, size):
        arr = self.lib.create_array(size, ctypes.sizeof(c_int))
        self.lib.fill_zeros_optimized(arr)
        return arr

    def arange(self, size):
        arr = self.lib.create_array(size, ctypes.sizeof(c_int))
        self.lib.fill_range_optimized(arr)
        return arr

    def sum(self, arr):
        return self.lib.sum_array_optimized(arr)

    def get_array_view(self, arr):
        return (c_int * arr.contents.size).from_address(arr.contents.data)

# Benchmark comparison
def benchmark():
    import numpy as np
    array_interface = FastArrayInterface()
    size = 10_000_000

    # Benchmark NumPy zeros
    start = time.perf_counter()
    np_arr = np.zeros(size, dtype=np.int32)
    np_time = time.perf_counter() - start
    print(f"NumPy zeros time: {np_time:.8f} seconds")

    # Benchmark our optimized zeros
    start = time.perf_counter()
    our_arr = array_interface.zeros(size)
    our_time = time.perf_counter() - start
    print(f"Our zeros time: {our_time:.8f} seconds")

    # Benchmark NumPy arange
    start = time.perf_counter()
    np_arr = np.arange(size, dtype=np.int32)
    np_time = time.perf_counter() - start
    print(f"NumPy arange time: {np_time:.8f} seconds")

    # Benchmark our optimized arange
    start = time.perf_counter()
    our_arr = array_interface.arange(size)
    our_time = time.perf_counter() - start
    print(f"Our arange time: {our_time:.8f} seconds")

if __name__ == "__main__":
    benchmark()
