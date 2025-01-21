import dis
import numpy as np

def example():
    arr = np.zeros((2, 2))
    print(arr)

# View the bytecode operations
dis.dis(example)

print(np.zeros.__module__)
print(np.zeros.__class__)
