import ctypes
import time

c_library = ctypes.CDLL('./my_c_library.so')

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
            
            return result_ptr  

        else:
            c_function.argtypes = [ctypes.c_int, ctypes.c_int]
            c_function.restype = ctypes.c_int
            result = c_function(*args)
        
        end_time = time.time()  
        elapsed_time = end_time - start_time
        print(f"Execution time for '{func_name}': {elapsed_time:.6f} seconds")
        
        return result
    return wrapper

@catch_return
def sum(a, b):
    pass

@catch_return
def create_array(size):
    return None  

result_sum = sum(3, 5)
print(f"Sum result: {result_sum}")

array_size = 10_000_000
result_array_ptr = create_array(array_size)
print(f"Pointer to array of zeros (size {array_size}): {result_array_ptr}")
