import numpy as np
from fast_cpp_caller import register_functions, execute_function

# Register all functions
register_functions()

# Call functions
result = execute_function("addInts", (5, 3))
print("Integer addition:", result)

# Make sure to use float arrays
arr1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
arr2 = np.array([4.0, 5.0, 6.0], dtype=np.float64)

# Print array info for debugging
print("Array 1 type:", arr1.dtype)
print("Array 2 type:", arr2.dtype)

result = execute_function("addArrays", (arr1, arr2))
print("Array addition:", result)
