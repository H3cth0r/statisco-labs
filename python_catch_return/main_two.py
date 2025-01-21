import ctypes
import time

c_library = ctypes.CDLL('./my_c_library.so')

class CArray:
    """
    A class to wrap and manage a C array returned by a shared library.
    """
    def __init__(self, ptr, size):
        self.ptr = ptr  
        self.size = size

    def __getitem__(self, index):
        if index < 0 or index >= self.size:
            raise IndexError("Array index out of range")
        return self.ptr[index]

    def __setitem__(self, index, value):
        if index < 0 or index >= self.size:
            raise IndexError("Array index out of range")
        self.ptr[index] = value

    def __len__(self):
        return self.size

    def __repr__(self):
        return f"CArray(size={self.size})"

    def to_list(self):
        """Convert the C array to a Python list (if needed)."""
        return [self.ptr[i] for i in range(self.size)]

    def free(self):
        """Free the memory allocated for the C array."""
        c_library.free_array(self.ptr)


def catch_return(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  

        func_name = func.__name__
        print(f"Calling function: {func_name}")

        c_function = getattr(c_library, func_name, None)
        if not c_function:
            raise AttributeError(f"C function '{func_name}' not found in the library")

        if func_name == 'create_array':
            c_function.argtypes = [ctypes.c_int]
            c_function.restype = ctypes.POINTER(ctypes.c_int)  

            size = args[0]
            result_ptr = c_function(size)
            if not result_ptr:
                raise MemoryError("Failed to allocate memory in C function.")
            
            end_time = time.time()
            print(f"Execution time for '{func_name}': {end_time - start_time:.6f} seconds")
            return CArray(result_ptr, size)

        else:
            c_function.argtypes = [ctypes.c_int, ctypes.c_int]
            c_function.restype = ctypes.c_int
            result = c_function(*args)

            end_time = time.time() 
            print(f"Execution time for '{func_name}': {end_time - start_time:.6f} seconds")
            return result

    return wrapper

@catch_return
def sum(a, b):
    pass

@catch_return
def create_array(size):
    pass

result_sum = sum(3, 5)
print(f"Sum result: {result_sum}")

array_size = 10_000_000
result_array = create_array(array_size)
print(f"Array created: {result_array}")

print(f"First element: {result_array[0]}")
result_array[0] = 42
print(f"Updated first element: {result_array[0]}")

result_array.free()
