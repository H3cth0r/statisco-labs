import arrco
import ctypes

if __name__ == "__main__":
    arr = arrco.array(200, ctypes.c_int)
    arr.set(4, 3)
    arr.set(5, 6)
    arr.set(0, 43)
    # arr.ones() 
    print("done")
    res = arr.get(4)
    print(arr.tolist())
    print(res)
    res = arr.get(90)
    print("="*40)
    print(arr)
