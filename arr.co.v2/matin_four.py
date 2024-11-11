import numpy as np
import time

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
def test_np():
    a = np.zeros(1000)

@timer
def test_py():
    a = [0] * 1000

if __name__ ==  "__main__":
    test_np()
    test_np()
    test_py()
    test_py()
