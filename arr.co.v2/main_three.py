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

class TestClass:
    __slots__ = 'arbol'
    def __init__(self, arbol):
        self.arbol = arbol
    def suma(a, b): return a + b

@timer
def alloc_class_time(arbol):
    tc = TestClass(arbol)

if __name__ == "__main__":
    TestClass({"hola":1})
    TestClass({"hola":1})
    alloc_class_time("gato")
    alloc_class_time(1)
    alloc_class_time([1,  3, 5])
