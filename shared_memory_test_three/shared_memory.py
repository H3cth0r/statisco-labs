import mmap
import os
import struct
import time
import numpy as np
from enum import IntEnum

class Operation(IntEnum):
    ADD = 0
    SUBTRACT = 1
    MULTIPLY = 2

class SharedMemoryManager:
    def __init__(self, filename="shared_math.dat", max_array_size=1000):
        self.filename = filename
        self.max_array_size = max_array_size
        
        self.header_format = '=ii??d'  # int32, int32, 2 bools, double for execution time
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
        
    def write_arrays(self, arr1, arr2, operation):
        if len(arr1) != len(arr2) or len(arr1) > self.max_array_size:
            raise ValueError(f"Arrays must be equal length and <= {self.max_array_size}")
        
        self.mm.seek(0)
        header = struct.pack(self.header_format, len(arr1), operation, False, False, 0.0)
        self.mm.write(header)
        
        header_size = struct.calcsize(self.header_format)
        array_size = len(arr1)
        
        self.mm.seek(header_size)
        arr1_bytes = struct.pack(f'{array_size}d', *arr1)
        self.mm.write(arr1_bytes)
        if array_size < self.max_array_size:
            self.mm.write(b'\x00' * ((self.max_array_size - array_size) * 8))
            
        self.mm.seek(header_size + (self.max_array_size * 8))
        arr2_bytes = struct.pack(f'{array_size}d', *arr2)
        self.mm.write(arr2_bytes)
        if array_size < self.max_array_size:
            self.mm.write(b'\x00' * ((self.max_array_size - array_size) * 8))
            
        self.mm.seek(0)
        header = struct.pack(self.header_format, array_size, operation, True, False, 0.0)
        self.mm.write(header)

    def read_result(self, size):
        self.mm.seek(0)
        
        header = self.mm.read(struct.calcsize(self.header_format))
        array_size, operation, ready, done, exec_time = struct.unpack(self.header_format, header)
        
        if not done:
            return None, False, 0.0
            
        self.mm.seek(struct.calcsize(self.header_format) + (self.max_array_size * 16))
        result_bytes = self.mm.read(size * 8)
        result = np.array(struct.unpack(f'{size}d', result_bytes))
        
        return result, True, exec_time

    def cleanup(self):
        self.mm.close()
        self.file.close()

def main():
    shm = SharedMemoryManager()
    array_size = 1000  # Larger size for better timing measurements
    num_calculations = 5
    operations = [Operation.ADD, Operation.SUBTRACT, Operation.MULTIPLY]
    
    print(f"Testing all operations with arrays of size {array_size}...")
    
    for op in operations:
        print(f"\nTesting {op.name} operation:")
        total_time = 0
        total_exec_time = 0
        
        for i in range(num_calculations):
            arr1 = np.random.rand(array_size)
            arr2 = np.random.rand(array_size)
            
            start_time = time.perf_counter()
            
            shm.write_arrays(arr1, arr2, op)
            
            retries = 0
            while True:
                result, done, exec_time = shm.read_result(array_size)
                if done:
                    break
                time.sleep(0.001)
                retries += 1
                if retries % 1000 == 0:
                    print(f"Waiting for result... (retry {retries})")
            
            duration = time.perf_counter() - start_time
            total_time += duration
            total_exec_time += exec_time
            
            if op == Operation.ADD:
                expected = arr1 + arr2
            elif op == Operation.SUBTRACT:
                expected = arr1 - arr2
            else:  # MULTIPLY
                expected = arr1 * arr2
                
            if not np.allclose(result, expected):
                print("Warning: Results don't match!")
                print(f"First few elements:")
                print(f"Expected: {expected[:3]}")
                print(f"Got: {result[:3]}")
            else:
                print(f"Iteration {i}: Success!")
                print(f"Total round-trip time: {duration*1000:.3f}ms")
                print(f"C++ execution time: {exec_time:.3f}ms")
        
        print(f"\n{op.name} Performance Statistics:")
        print(f"Average round-trip time: {(total_time/num_calculations)*1000:.3f}ms")
        print(f"Average C++ execution time: {(total_exec_time/num_calculations):.3f}ms")
        print(f"Operations per second: {num_calculations/total_time:.2f}")
    
    shm.cleanup()

if __name__ == "__main__":
    main()
