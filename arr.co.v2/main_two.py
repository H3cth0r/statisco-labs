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

class One:
    def __init__(self): pass
    def sum(self): return 1 + 1
class _Two:
    def __init__(self):pass
    def sum(self): return 1 + 1
Two = _Two()

@timer
def test_one():
    one = One().sum()
@timer
def test_two():
    one = Two.sum()
if __name__ == "__main__":
    test_one()
    test_two()
