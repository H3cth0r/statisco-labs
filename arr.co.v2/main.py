import time
import MallocAllocator 
import numpy as np

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Start the timer
        result = func(*args, **kwargs)  # Call the function
        end_time = time.time()  # End the timer
        duration = end_time - start_time  # Calculate duration
        print(f"Function '{func.__name__}' took {duration:.8f} seconds to complete.")
        return result
    return wrapper

@timer
def test():pass
@timer
def test_allocation(n):
    a = MallocAllocator.alloc(4)
    MallocAllocator.copyin(a, memoryview(bytearray([n, 0, 0, 0])))

@timer
def test_numpy(n): a = np.array([n, 0, 0, 0])


# b = MallocAllocator.alloc(1000)
# MallocAllocator.copyin(b, memoryview(bytearray([3, 0, 0, 0])))
test()
# test_numpy()
test_allocation(34)
test_allocation(34)
test_allocation(34)
test_numpy(34)
test_numpy(34)
test_numpy(34)
