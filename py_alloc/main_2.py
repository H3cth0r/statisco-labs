from source_code import dtypes
from ops_clang import Buffer

if __name__ == "__main__":
    DEVICE = "CLANG"
    out = Buffer(DEVICE, 1, dtypes.int32).allocate()
