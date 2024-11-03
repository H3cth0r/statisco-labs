from source_code import MallocAllocator
from ops_clang import ClangCompiler, ClangProgram

if __name__ == "__main__":
    # Allocate buffer
    out = MallocAllocator.alloc(4)
    a = MallocAllocator.alloc(4)
    b = MallocAllocator.alloc(4)
    # print(a)

    MallocAllocator.copyin(a, memoryview(bytearray([2, 0, 0, 0])))
    MallocAllocator.copyin(b, memoryview(bytearray([3, 0, 0, 0])))

    lib = ClangCompiler().compile("void add(int *out, int *a, int *b) { out[0] = a[0] + b[0]; }")
    # print(lib)

    fxn = ClangProgram("add", lib)
    fxn(out, a, b)
    print(val := MallocAllocator.as_buffer(out).cast("I").tolist()[0])
    print(out)
    assert val == 5
