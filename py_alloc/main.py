from source_code import MallocAllocator

if __name__ == "__main__":
    # Allocate buffer
    # out = MallocAllocator.alloc(4)
    a = MallocAllocator.alloc(4)
    print(a)
    MallocAllocator.copyin(a, memoryview(bytearray([2, 0, 0, 0])))
    print(a)
