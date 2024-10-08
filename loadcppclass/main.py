import ctypes

lib = ctypes.CDLL('./point.so')

lib.create_point.restype    = ctypes.c_void_p
lib.create_point.argtypes   = [ctypes.c_int, ctypes.c_int]

lib.delete_point.argtypes   = [ctypes.c_void_p]

lib.print_point.argtypes    = [ctypes.c_void_p]

lib.set_x.argtypes          = [ctypes.c_void_p, ctypes.c_int]
lib.set_y.argtypes          = [ctypes.c_void_p, ctypes.c_int]

lib.get_x.argtypes          = [ctypes.c_void_p]
lib.get_x.restype           = ctypes.c_int

lib.get_y.argtypes          = [ctypes.c_void_p]
lib.get_y.restype           = ctypes.c_int

if __name__ == "__main__":
    p = lib.create_point(10, 20)
    lib.print_point(p)
