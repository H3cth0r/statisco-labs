from warm_up_module import TestClass
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
def test():pass
@timer
def test_allocation():
    obj = TestClass("gato")
    obj.suma(3, 4)

if __name__ == "__main__":
    test()
    test_allocation()
