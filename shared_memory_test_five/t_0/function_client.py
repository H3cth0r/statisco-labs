import mmap
import ctypes
import time
import numpy as np
from typing import Any

class CommandBlock(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("function_name", ctypes.c_char * 64),
        ("request_pending", ctypes.c_bool),
        ("response_ready", ctypes.c_bool),
        ("should_exit", ctypes.c_bool),
    ]

class FunctionClient:
    def __init__(self):
        # Open command block file
        self.cmd_file = open("command_block.dat", "r+b")
        self.cmd_mm = mmap.mmap(self.cmd_file.fileno(), 0)
        self.cmd = CommandBlock.from_buffer(self.cmd_mm)
        
        # Open data file
        self.data_file = open("functions.dat", "r+b")
        self.data_mm = mmap.mmap(self.data_file.fileno(), 0)
    
    def execute_function(self, name: str, *args: Any) -> Any:
        try:
            # Set function name
            self.cmd.function_name = name.encode('utf-8')
            
            # Reset the data memory offset
            offset = 0
            
            # Write arguments to shared memory
            for arg in args:
                if isinstance(arg, int):
                    self.data_mm[offset:offset + ctypes.sizeof(ctypes.c_int)] = \
                        (ctypes.c_int(arg).value).to_bytes(ctypes.sizeof(ctypes.c_int), byteorder='little')
                    offset += ctypes.sizeof(ctypes.c_int)
                elif isinstance(arg, float):
                    self.data_mm[offset:offset + ctypes.sizeof(ctypes.c_double)] = \
                        (ctypes.c_double(arg).value).to_bytes(ctypes.sizeof(ctypes.c_double), byteorder='little')
                    offset += ctypes.sizeof(ctypes.c_double)
                elif isinstance(arg, np.ndarray):
                    arr_bytes = arg.tobytes()
                    self.data_mm[offset:offset + len(arr_bytes)] = arr_bytes
                    offset += len(arr_bytes)
                else:
                    raise ValueError(f"Unsupported argument type: {type(arg)}")
            
            # Request execution
            self.cmd.request_pending = True
            self.cmd_mm.flush()
            
            # Wait for response
            while self.cmd.request_pending:
                time.sleep(0.01)
                self.cmd_mm.seek(0)
                self.cmd = CommandBlock.from_buffer(self.cmd_mm)
            
            # Get result from shared memory
            result = None
            if self.cmd.response_ready:
                # Dynamically determine the return type based on the function name or metadata
                if name == "addInts":
                    result = int.from_bytes(self.data_mm[:ctypes.sizeof(ctypes.c_int)], byteorder='little')
                elif name == "addArrays":
                    result = np.frombuffer(self.data_mm, dtype=np.int32, count=3)
                else:
                    # For other functions, assume the result is a single integer (default)
                    result = int.from_bytes(self.data_mm[:ctypes.sizeof(ctypes.c_int)], byteorder='little')
                
                self.cmd.response_ready = False
                self.cmd_mm.flush()
            
            return result
        
        except Exception as e:
            print(f"Error executing function: {e}")
            return None
    
    def close(self):
        try:
            self.cmd.should_exit = True
            self.cmd_mm.flush()
            self.cmd_mm.close()
            self.cmd_file.close()
            self.data_mm.close()
            self.data_file.close()
        except Exception as e:
            print(f"Error during cleanup: {e}")

# Example usage
if __name__ == "__main__":
    client = FunctionClient()
    
    try:
        # Example function calls
        result = client.execute_function("addInts", 5, 3)
        print(f"5 + 3 = {result}")
        
        arr1 = np.array([1, 2, 3], dtype=np.int32)
        arr2 = np.array([4, 5, 6], dtype=np.int32)
        result = client.execute_function("addArrays", arr1, arr2)
        print(f"Array sum: {result}")
        
    finally:
        client.close()
