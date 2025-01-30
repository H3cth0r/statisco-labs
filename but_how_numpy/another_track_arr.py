import numpy as np
import inspect
import sys
from types import ModuleType, FunctionType
import ctypes

def trace_numpy_function(func_name):
    """
    Traces the call hierarchy and implementation details of a NumPy function
    
    Parameters:
    func_name (str): The NumPy function to trace (e.g., 'array', 'zeros')
    """
    def get_source_if_available(obj):
        try:
            return inspect.getsource(obj)
        except (TypeError, OSError):
            return "Source code not available (possibly implemented in C)"

    def trace_object(obj, depth=0, visited=None):
        if visited is None:
            visited = set()
            
        # Avoid circular references
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)
        
        indent = "  " * depth
        
        # Print basic object information
        print(f"{indent}Object: {obj.__name__ if hasattr(obj, '__name__') else type(obj).__name__}")
        print(f"{indent}Type: {type(obj)}")
        print(f"{indent}Module: {getattr(obj, '__module__', 'Unknown')}")
        
        # Get source code or implementation details
        source = get_source_if_available(obj)
        print(f"{indent}Source:\n{source}\n")
        
        # Look for C extensions
        if hasattr(obj, '__doc__') and 'C-API' in str(obj.__doc__):
            print(f"{indent}This appears to be implemented in C")
            
        # For modules, explore relevant attributes
        if isinstance(obj, ModuleType):
            for name, attr in inspect.getmembers(obj):
                if name.startswith('_'):
                    continue
                if isinstance(attr, (FunctionType, type)):
                    trace_object(attr, depth + 1, visited)

    # Start tracing from the NumPy function
    target_func = getattr(np, func_name)
    print(f"Tracing NumPy function: {func_name}\n")
    trace_object(target_func)
    
    # Try to find C extension information
    try:
        numpy_dll = ctypes.CDLL(np.__file__)
        print(f"\nC Extension Information:")
        print(f"NumPy DLL location: {np.__file__}")
        # List available C functions (if possible)
        for name, func in inspect.getmembers(numpy_dll):
            if not name.startswith('_'):
                print(f"C function: {name}")
    except Exception as e:
        print(f"\nCouldn't load C extension information: {e}")

# Example usage functions
def analyze_array_creation():
    """Analyze np.array creation"""
    print("=== Analyzing np.array creation ===")
    trace_numpy_function('array')
    
    # Create a small array and examine its memory layout
    arr = np.array([1, 2, 3])
    print("\nArray Memory Information:")
    print(f"Data address: {arr.__array_interface__['data'][0]}")
    print(f"Strides: {arr.strides}")
    print(f"Shape: {arr.shape}")
    print(f"Dtype: {arr.dtype}")

def analyze_zeros_creation():
    """Analyze np.zeros creation"""
    print("\n=== Analyzing np.zeros creation ===")
    trace_numpy_function('zeros')
    
    # Create a zeros array and examine its memory layout
    arr = np.zeros((3, 3))
    print("\nZeros Array Memory Information:")
    print(f"Data address: {arr.__array_interface__['data'][0]}")
    print(f"Strides: {arr.strides}")
    print(f"Shape: {arr.shape}")
    print(f"Dtype: {arr.dtype}")

if __name__ == "__main__":
    analyze_array_creation()
    analyze_zeros_creation()
