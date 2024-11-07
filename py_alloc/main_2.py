from source_code import dtypes
from ops_clang import Buffer
import struct

if __name__ == "__main__":
    DEVICE = "CLANG"

    # allocate some buffers and load in values
    out = Buffer(DEVICE, 1, dtypes.int32).allocate()
    a = Buffer(DEVICE, 1, dtypes.int32).allocate().copyin(memoryview(bytearray(struct.pack("I", 2))))
    b = Buffer(DEVICE, 1, dtypes.int32).allocate().copyin(memoryview(bytearray(struct.pack("I", 3))))

    # describe the computation
    print("finish")
