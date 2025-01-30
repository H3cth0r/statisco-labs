import mmap
import ctypes
import timeit
import statistics

def create_executable_memory(code_bytes):
    """Creates executable memory and writes the given machine code into it."""
    size = len(code_bytes)
    
    # Allocate executable memory with correct flags
    mem = mmap.mmap(-1, size, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, 
                    mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC)
    
    # Write machine code into allocated memory
    mem.write(code_bytes)
    
    # Move pointer back to start
    mem.seek(0)
    
    return mem

# Machine code for add(x, y) => x + y (x86-64 System V calling convention)
ADD_FUNCTION = bytes([
    0x89, 0xf8,     # mov eax, edi  ; Move first argument into eax
    0x01, 0xf0,     # add eax, esi  ; Add second argument
    0xc3            # ret           ; Return result in eax
])

# Machine code for multiply(x, y) => x * y (x86-64 System V calling convention)
MULTIPLY_FUNCTION = bytes([
    0x89, 0xf8,     # mov eax, edi  ; Move first argument into eax
    0x0f, 0xaf, 0xc6,  # imul eax, esi ; Multiply by second argument
    0xc3            # ret           ; Return result in eax
])

# Create executable memory regions
add_mem = create_executable_memory(ADD_FUNCTION)
mul_mem = create_executable_memory(MULTIPLY_FUNCTION)

# Get the pointer to the allocated memory
add_addr = ctypes.addressof(ctypes.c_char.from_buffer(add_mem))
mul_addr = ctypes.addressof(ctypes.c_char.from_buffer(mul_mem))

# Create callable function objects
add_func = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int)(add_addr)
mul_func = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int)(mul_addr)

# Test functions
print("Testing direct machine code execution:")
result1 = add_func(5, 3)
result2 = mul_func(5, 3)
print(f"add(5, 3) = {result1}")
print(f"multiply(5, 3) = {result2}")

# Function to time execution
def time_function(func, *args, number=1000000):
    """Time a function call over multiple iterations."""
    stmt = lambda: func(*args)
    times = timeit.repeat(stmt, number=number, repeat=5)
    avg_time = statistics.mean(times) / number * 1e6  # Convert to microseconds
    return avg_time

# Timing tests
add_time = time_function(add_func, 5, 3)
mul_time = time_function(mul_func, 5, 3)

print(f"\nAverage execution times:")
print(f"Add function: {add_time:.3f} microseconds per call")
print(f"Multiply function: {mul_time:.3f} microseconds per call")

# Cleanup
add_mem.close()
mul_mem.close()
