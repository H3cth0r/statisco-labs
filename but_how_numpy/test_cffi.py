# First install cffi: pip install cffi
from cffi import FFI
import timeit
import statistics

ffi = FFI()

# Define the C code
ffi.cdef("""
    int add(int a, int b);
    float multiply(float x, float y);
""")

# Provide the implementation
ffi.set_source("_example",
    """
    int add(int a, int b) {
        return a + b;
    }
    
    float multiply(float x, float y) {
        return x * y;
    }
    """)

# Build the C extension
if __name__ == "__main__":
    ffi.compile()

# Usage after compilation:
from _example import ffi, lib

def time_function(func, *args, number=1000000):
    """Time a function call over multiple iterations"""
    stmt = lambda: func(*args)
    times = timeit.repeat(stmt, number=number, repeat=5)
    avg_time = statistics.mean(times) / number * 1e6  # Convert to microseconds
    return avg_time

# Test add function
add_time = time_function(lib.add, 5, 3)
add_result = lib.add(5, 3)
print(f"add(5, 3) = {add_result}")
print(f"Average execution time: {add_time:.2f} microseconds per call\n")

# Test multiply function
multiply_time = time_function(lib.multiply, 2.5, 3)
multiply_result = lib.multiply(2.5, 3)
print(f"multiply(2.5, 3) = {multiply_result}")
print(f"Average execution time: {multiply_time:.2f} microseconds per call")

# Compare with pure Python
def py_add(a, b):
    return a + b

def py_multiply(x, y):
    return x * y

py_add_time = time_function(py_add, 5, 3)
py_multiply_time = time_function(py_multiply, 2.5, 3)

print("\nComparison with Python:")
print(f"C add vs Python add: {py_add_time/add_time:.1f}x slower")
print(f"C multiply vs Python multiply: {py_multiply_time/multiply_time:.1f}x slower")
