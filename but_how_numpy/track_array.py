import numpy as np
import inspect
import sys
from functools import wraps
import traceback

def trace_function_calls(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get the current frame and its depth
        current_frame = sys._getframe()
        depth = 0
        while current_frame.f_back:
            current_frame = current_frame.f_back
            depth += 1
        
        # Create indentation based on stack depth
        indent = "  " * (depth - 3)  # Adjust base indentation
        
        # Print function entry
        print(f"{indent}→ Entering {func.__module__}.{func.__name__}")
        print(f"{indent}  Arguments: {args}, {kwargs}")
        
        # Get source code if available
        try:
            source = inspect.getsource(func)
            print(f"{indent}  Source:\n{indent}    " + source.replace('\n', f'\n{indent}    '))
        except Exception:
            print(f"{indent}  Source code not available")
        
        # Call the original function
        result = func(*args, **kwargs)
        
        # Print function exit and result type
        print(f"{indent}← Exiting {func.__module__}.{func.__name__}")
        print(f"{indent}  Returns: {type(result)}")
        print(f"{indent}  Result shape/type: {result.shape if hasattr(result, 'shape') else 'N/A'}")
        
        return result
    return wrapper

# Patch numpy array creation functions
np.array = trace_function_calls(np.array)
np.zeros = trace_function_calls(np.zeros)

def investigate_numpy_creation():
    print("Creating array using np.array:")
    arr1 = np.array([1, 2, 3])
    print("\nCreating array using np.zeros:")
    arr2 = np.zeros((2, 3))
    
    # Inspect the created arrays
    print("\nArray 1 details:")
    print(f"Type: {type(arr1)}")
    print(f"Shape: {arr1.shape}")
    print(f"Data type: {arr1.dtype}")
    print(f"Base object: {arr1.base}")
    
    print("\nArray 2 details:")
    print(f"Type: {type(arr2)}")
    print(f"Shape: {arr2.shape}")
    print(f"Data type: {arr2.dtype}")
    print(f"Base object: {arr2.base}")
    
    # Get the MRO (Method Resolution Order) for ndarray
    print("\nClass hierarchy for numpy.ndarray:")
    for cls in np.ndarray.__mro__:
        print(f"- {cls.__module__}.{cls.__name__}")

if __name__ == "__main__":
    investigate_numpy_creation()
