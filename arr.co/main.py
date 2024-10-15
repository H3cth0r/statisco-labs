import arrco
import ctypes

if __name__ == "__main__":
    arr = arrco.array(110, ctypes.c_int)
    arr.set(4, 3)
    res = arr.get(4)
    print(res)
