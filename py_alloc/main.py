from source_code import MallocAllocator
from ops_clang import ClangCompiler

if __name__ == "__main__":
    # Allocate buffer
    # out = MallocAllocator.alloc(4)
    a = MallocAllocator.alloc(4)
    print(a)
    MallocAllocator.copyin(a, memoryview(bytearray([2, 0, 0, 0])))
    print(a)
    lib = ClangCompiler().compile("void add(int *out, int *a, int *b) { out[0] = a[0] + b[0]; }")
