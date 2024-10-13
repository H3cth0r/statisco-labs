from vectorarray import VectorArray
import ctypes

DTYPE_INT = 0
DTYPE_FLOAT = 1
DTYPE_DOUBLE = 2
DTYPE_CHAR = 3

if __name__ == "__main__":
    arr = VectorArray(100, ctypes.c_int)
    print("="*30)
    arr.set(0, 3)
    print("="*30)
    res = arr.get(0)
    print("="*30)
    print(res)
    print("Howdy")
