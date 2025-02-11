import mmap
import ctypes
import time
import numpy as np
from typing import Any

class FixedArray(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("data", ctypes.c_int * 1024),
        ("size", ctypes.c_size_t)
    ]

class CommandBlock(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("function_name", ctypes.c_char * 64),
        ("request_pending", ctypes.c_bool),
        ("response_ready", ctypes.c_bool),
        ("should_exit", ctypes.c_bool),
    ]

class FunctionMetadata(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("name", ctypes.c_char * 64),
        ("num_args", ctypes.c_size_t),
        ("return_offset", ctypes.c_size_t),
        ("return_size", ctypes.c_size_t),
        ("arg_offsets", ctypes.c_size_t * 10),
        ("arg_sizes", ctypes.c_size_t * 10),
    ]

class SharedMemoryHeader(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("num_functions", ctypes.c_size_t),
        ("data_start_offset", ctypes.c_size_t),
        ("functions", FunctionMetadata * 100),
    ]

class FunctionClient:
    def __init__(self):
        self.cmd_file = open("command_block.dat", "r+b")
        self.cmd_mm = mmap.mmap(self.cmd_file.fileno(), 0)
        self.cmd = CommandBlock.from_buffer(self.cmd_mm)
        
        self.data_file = open("functions.dat", "r+b")
        self.data_mm = mmap.mmap(self.data_file.fileno(), 0)
        self.header = SharedMemoryHeader.from_buffer(self.data_mm)
        
        self.functions = {}
        for i in range(self.header.num_functions):
            func = self.header.functions[i]
            self.functions[func.name.decode('utf-8').strip('\0')] = func
    
    def execute_function(self, name: str, *args: Any) -> Any:
        if name not in self.functions:
            raise ValueError(f"Function {name} not found")
            
        metadata = self.functions[name]
        if len(args) != metadata.num_args:
            raise ValueError(f"Expected {metadata.num_args} arguments, got {len(args)}")
        
        # Write arguments to shared memory
        for i, arg in enumerate(args):
            offset = metadata.arg_offsets[i]
            size = metadata.arg_sizes[i]
            
            if isinstance(arg, (int, float)):
                self.data_mm[offset:offset + size] = arg.to_bytes(size, byteorder='little', signed=True)
            elif isinstance(arg, (list, tuple, np.ndarray)):
                # Convert to FixedArray
                fixed_array = FixedArray()
                arr = np.array(arg, dtype=np.int32)
                fixed_array.size = len(arr)
                for j, val in enumerate(arr):
                    fixed_array.data[j] = val
                self.data_mm[offset:offset + size] = bytes(fixed_array)
            else:
                raise ValueError(f"Unsupported argument type: {type(arg)}")
        
        # request execution
        self.cmd.function_name = name.encode('utf-8')
        self.cmd.request_pending = True
        self.cmd_mm.flush()
        
        # wait response
        while self.cmd.request_pending:
            time.sleep(0.00005)
            self.cmd_mm.seek(0)
            self.cmd = CommandBlock.from_buffer(self.cmd_mm)
        
        # get result
        if self.cmd.response_ready:
            if metadata.return_size == ctypes.sizeof(ctypes.c_int):
                result_bytes = self.data_mm[metadata.return_offset:metadata.return_offset + metadata.return_size]
                return int.from_bytes(result_bytes, byteorder='little', signed=True)
            elif metadata.return_size == ctypes.sizeof(FixedArray):
                result = FixedArray.from_buffer(self.data_mm, metadata.return_offset)
                return [result.data[i] for i in range(result.size)]
        
        return None
    
    def close(self):
        try:
            self.cmd.should_exit = True
            self.cmd_mm.flush()
            self.cmd_file.close()
            self.data_file.close()
        except Exception as e:
            print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    client = FunctionClient()
    
    try:
        t_start = time.perf_counter()
        result = client.execute_function("addInts", 5, 3)
        t_total = time.perf_counter() - t_start
        print(f"5 + 3 = {result}")
        print(f"elapsed time {t_total}")

        t_start = time.perf_counter()
        result = client.execute_function("addInts", 4, 7)
        t_total = time.perf_counter() - t_start
        print(f"4 + 7 = {result}")
        print(f"elapsed time {t_total}")

        t_start = time.perf_counter()
        result = client.execute_function("addMoreInts", 5, 3, 4)
        t_total = time.perf_counter() - t_start
        print(f"5 + 3 + 4 = {result}")
        print(f"elapsed time {t_total}")
        
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])
        t_start = time.perf_counter()
        result = client.execute_function("addArrays", arr1, arr2)
        t_total = time.perf_counter() - t_start
        print(f"Array sum: {result}")
        print(f"elapsed time {t_total}")

        t_start = time.perf_counter()
        result = arr1 - arr2
        t_total = time.perf_counter() - t_start
        print(f"Array sum: {result}")
        print(f"Elapsed time: {t_total:.10f} seconds")
        
    finally:
        client.close()
