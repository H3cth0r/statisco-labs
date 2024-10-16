import arrco
import ctypes

if __name__ == "__main__":
    arr = arrco.array(110, ctypes.c_int)
    # arr.set(4, 3)
    # arr.set(5, 6)
    arr.ones() 
    print("done")
    res = arr.get(4)
    print(res)
    res = arr.get(90)
    print(res)
