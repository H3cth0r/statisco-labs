import time

class WithInit:
    def __init__(self):
        self.one = "hola"
class WithNew:
    def __new__():
        one = "hola"

def timed_method(func):
  def wrapper(*args, **kwargs):
    start_time      = time.time()  # Record start time
    result          = func(*args, **kwargs)  # Execute the original method
    end_time        = time.time()  # Record end time
    execution_time  = end_time - start_time
    print(f"{func.__name__} took {execution_time:.6f} seconds to execute.")
    return result
  return wrapper

@timed_method
def first_one():
    res = WithInit()

@timed_method
def second_one():
    res = WithNew.__new__()

if __name__ == "__main__":
    first_one()
    first_one()
    second_one()
