import ctypes
import time

# Load the shared library
c_library = ctypes.CDLL('./my_c_library.so')

def catch_return(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Start timing
        
        func_name = func.__name__
        print(f"Calling function: {func_name}")
        
        # Get the corresponding C function
        c_function = getattr(c_library, func_name, None)
        if not c_function:
            raise AttributeError(f"C function '{func_name}' not found in the library")
        
        # Special handling for 'create_array' (returns a pointer to an array)
        if func_name == 'create_array':
            c_function.argtypes = [ctypes.c_int]
            c_function.restype = ctypes.POINTER(ctypes.c_int)  # Return a pointer to int
            
            # Call the C function
            size = args[0]
            result_ptr = c_function(size)
            if not result_ptr:
                raise MemoryError("Failed to allocate memory in C function.")
            
            # Return the pointer itself, not converted to a Python list
            return result_ptr  # We return the pointer directly

        else:
            # Default handling for other functions (e.g., 'sum')
            c_function.argtypes = [ctypes.c_int, ctypes.c_int]
            c_function.restype = ctypes.c_int
            result = c_function(*args)
        
        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time
        print(f"Execution time for '{func_name}': {elapsed_time:.6f} seconds")
        
        return result
    return wrapper

# Apply the decorator
@catch_return
def sum(a, b):
    return 0  # Placeholder

@catch_return
def create_array(size):
    return None  # Placeholder

# Example usage
result_sum = sum(3, 5)
print(f"Sum result: {result_sum}")

array_size = 10_000_000
result_array_ptr = create_array(array_size)
print(f"Pointer to array of zeros (size {array_size}): {result_array_ptr}")
