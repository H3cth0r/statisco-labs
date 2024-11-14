import pyrust_alloc
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
def test_rust():
    current_time = pyrust_alloc.get_current_time()

@timer
def test_py():
    format_string = "%Y-%m-%d %H:%M:%S"
    return time.strftime(format_string)

test_py()
test_rust()
test_rust()
