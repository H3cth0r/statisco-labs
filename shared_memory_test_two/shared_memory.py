import mmap
import os
import struct
import time
import numpy as np

class SharedMemoryManager:
    def __init__(self, filename="shared_math.dat", max_array_size=1000):
        self.filename = filename
        self.max_array_size = max_array_size
        
        # Match C++ struct exactly
        self.header_format = '=i??'  # int32 + 2 bools, = enforces native alignment
        self.array_format = f'{max_array_size}d'  # array of doubles
        self.struct_size = (
            struct.calcsize(self.header_format) +  # Header
            (max_array_size * 8 * 3)  # 3 arrays of doubles (8 bytes each)
        )
        
        if not os.path.exists(self.filename):
            with open(self.filename, "wb") as f:
                f.write(b'\x00' * self.struct_size)
        
        self.file = open(self.filename, "r+b")
        self.mm = mmap.mmap(self.file.fileno(), self.struct_size)
        
    def write_arrays(self, arr1, arr2):
        if len(arr1) != len(arr2) or len(arr1) > self.max_array_size:
            raise ValueError(f"Arrays must be equal length and <= {self.max_array_size}")
        
        # Reset flags first
        self.mm.seek(0)
        header = struct.pack(self.header_format, len(arr1), False, False)
        self.mm.write(header)
        
        # Write arrays
        header_size = struct.calcsize(self.header_format)
        array_size = len(arr1)
        
        # Write first array
        self.mm.seek(header_size)
        arr1_bytes = struct.pack(f'{array_size}d', *arr1)
        self.mm.write(arr1_bytes)
        if array_size < self.max_array_size:
            self.mm.write(b'\x00' * ((self.max_array_size - array_size) * 8))
            
        # Write second array
        self.mm.seek(header_size + (self.max_array_size * 8))
        arr2_bytes = struct.pack(f'{array_size}d', *arr2)
        self.mm.write(arr2_bytes)
        if array_size < self.max_array_size:
            self.mm.write(b'\x00' * ((self.max_array_size - array_size) * 8))
            
        # Set ready flag
        self.mm.seek(0)
        header = struct.pack(self.header_format, array_size, True, False)
        self.mm.write(header)

    def read_result(self, size):
        self.mm.seek(0)
        
        # Read header
        header = self.mm.read(struct.calcsize(self.header_format))
        array_size, ready, done = struct.unpack(self.header_format, header)
        
        if not done:
            return None, False
            
        # Read result array
        self.mm.seek(struct.calcsize(self.header_format) + (self.max_array_size * 16))  # Skip header and input arrays
        result_bytes = self.mm.read(size * 8)
        result = np.array(struct.unpack(f'{size}d', result_bytes))
        
        return result, True

    def cleanup(self):
        self.mm.close()
        self.file.close()

def main():
    shm = SharedMemoryManager()
    array_size = 100  # Smaller size for testing
    num_calculations = 10
    
    print(f"Starting {num_calculations} calculations with arrays of size {array_size}...")
    total_time = 0
    
    for i in range(num_calculations):
        arr1 = np.random.rand(array_size)
        arr2 = np.random.rand(array_size)
        
        start_time = time.perf_counter()
        
        # Write arrays and wait for result
        shm.write_arrays(arr1, arr2)
        
        retries = 0
        while True:
            result, done = shm.read_result(array_size)
            if done:
                break
            time.sleep(0.001)  # 1ms sleep
            retries += 1
            if retries % 1000 == 0:  # Print every 1000 retries
                print(f"Waiting for result... (retry {retries})")
        
        duration = time.perf_counter() - start_time
        total_time += duration
        
        # Verify result
        expected = arr1 + arr2
        if not np.allclose(result, expected):
            print("Warning: Results don't match!")
            print(f"First few elements:")
            print(f"Expected: {expected[:3]}")
            print(f"Got: {result[:3]}")
        else:
            print(f"Iteration {i}: Success! Processing time: {duration*1000:.3f}ms")
    
    # Print statistics
    avg_time = (total_time / num_calculations) * 1000
    ops_per_second = num_calculations / total_time
    elements_per_second = (array_size * num_calculations) / total_time
    
    print("\nPerformance Statistics:")
    print(f"Total calculations: {num_calculations}")
    print(f"Array size: {array_size}")
    print(f"Total time: {total_time:.3f} seconds")
    print(f"Average time per calculation: {avg_time:.3f} ms")
    print(f"Calculations per second: {ops_per_second:.2f}")
    print(f"Elements processed per second: {elements_per_second:.2f}")
    
    shm.cleanup()

if __name__ == "__main__":
    main()
