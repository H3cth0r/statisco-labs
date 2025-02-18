import numpy as np
import time
from fast_cpp_caller import execute_function, addInts

def addMoreInts(*args): return execute_function("addMoreInts", args)


# Call functions
timer = time.time()
result = execute_function("addInts", (5, 3))
# result = addInts(5, 3)
timer = time.time() - timer
print(f"timer: {timer:.16f}")  # Print full decimal notation
print("Integer addition:", result)
print("="*15)

timer = time.time()
result = execute_function("addInts", (5, 3))
# result = addInts(5, 3)
timer = time.time() - timer
print(f"timer: {timer:.16f}")  # Print full decimal notation
print("Integer addition:", result)
print("="*15)

timer = time.time()
# result = execute_function("addInts", (5, 3))
result = addInts(5, 3)
timer = time.time() - timer
print(f"timer: {timer:.16f}")  # Print full decimal notation
print("Integer addition:", result)
print("="*15)

timer = time.time()
# result = execute_function("addInts", (5, 3))
result = addInts(5, 3)
timer = time.time() - timer
print(f"timer: {timer:.16f}")  # Print full decimal notation
print("Integer addition:", result)
print("="*15)


# Make sure to use float arrays
arr1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
arr2 = np.array([4.0, 5.0, 6.0], dtype=np.float64)

# Print array info for debugging
print("Array 1 type:", arr1.dtype)
print("Array 2 type:", arr2.dtype)

timer = time.time()
result = execute_function("addArrays", (arr1, arr2))
timer = time.time() - timer
print(f"timer: {timer:.16f}")  # Print full decimal notation
print("Array addition:", result)

timer = time.time()
result = arr1 + arr2
timer = time.time() - timer
print(f"timer: {timer:.16f}")  # Print full decimal notation
print("Array addition:", result)

print(type(result))
