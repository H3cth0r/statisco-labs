from source_code import MallocAllocator
import time
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
def test_allocation():
    a = MallocAllocator.alloc(4)
    MallocAllocator.copyin(a, memoryview(bytearray([2, 0, 0, 0])))
@timer
def test_np_allocation():
    a = np.array([2, 0, 0, 0])

if __name__ == "__main__":
    print("asd")
    test_allocation()
    test_np_allocation()
    test_allocation()
