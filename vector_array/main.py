import time

from VectorArray import VectorArray, zeros
import numpy as np

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
def load_single_vector(size):
    vec_a = VectorArray.zeros(size)
    return vec_a

@timed_method
def load_single_numpy_array(size):
    vec_a = np.zeros(size, dtype=np.double)
    return vec_a

@timed_method
def load_single_vector_method(size):
    vec_a, buffer= zeros(size)
    return vec_a, buffer

va = {
        "zeros" : zeros
}

@timed_method
def load_from_dict(size):
    vec_a, buffer= va["zeros"](size)
    return vec_a, buffer

if __name__ == "__main__":
    size    = 10000000
    res_2 = load_single_numpy_array(size)
    res_1 = load_single_vector(size)
    res_3, buffer = load_single_vector_method(size)
    res_4, buffer1 = load_from_dict(size)

    print(res_1[:10])
    print(res_2[:10])
    print(res_3[:10])
    print(res_4[:10])


    del res_3
    buffer.close()

    del res_4
    buffer1.close()
