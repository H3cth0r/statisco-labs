import mmap
import os
import struct
import time
import random
from datetime import datetime

class SharedMemoryManager:
    def __init__(self, filename="shared_math.dat"):
        self.filename = filename
        self.struct_format = 'ddd??'
        self.struct_size = struct.calcsize(self.struct_format)

        if not os.path.exists(self.filename):
            with open(self.filename, "wb") as f:
                f.write(b'\x00' * self.struct_size)

        self.file = open(self.filename, "r+b")
        self.mm = mmap.mmap(self.file.fileno(), self.struct_size)

    def write_data(self, num1, num2):
        self.mm.seek(0)
        data = struct.pack(self.struct_format, num1, num2, 0.0, True, False)
        self.mm.write(data)
    def read_result(self):
        self.mm.seek(0)
        data = self.mm.read(self.struct_size)
        num1, num2, result, ready, done = struct.unpack(self.struct_format, data)
        return result, done
    def cleanup(self):
        self.mm.close()
        self.file.close()

def main():
    shm = SharedMemoryManager()
    num_calculations = 1000
    total_time = 0

    print(f"Starting {num_calculations} calculations...")

    for i in range(num_calculations):
        num1 = random.uniform(1, 100)
        num2 = random.uniform(1, 100)
        
        start_time = time.perf_counter()
        
        shm.write_data(num1, num2)
        
        while True:
            result, done = shm.read_result()
            if done:
                break
            time.sleep(0.00005)  # Small sleep to prevent CPU hogging
        
        duration = time.perf_counter() - start_time
        total_time += duration
        
        if i % 100 == 0:  # Print progress every 100 calculations
            print(f"Calculation {i}: {num1} + {num2} = {result} (took {duration*1000:.3f}ms)")
    
    avg_time = (total_time / num_calculations) * 1000  # Convert to milliseconds
    ops_per_second = num_calculations / total_time
    
    print("\nPerformance Statistics:")
    print(f"Total calculations: {num_calculations}")
    print(f"Total time: {total_time:.3f} seconds")
    print(f"Average time per calculation: {avg_time:.3f} ms")
    print(f"Operations per second: {ops_per_second:.2f}")
    
    shm.cleanup()

if __name__ == "__main__":
    main()
