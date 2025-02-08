import mmap
import os
import struct
import time
import numpy as np
from enum import Enum
from typing import Any, Dict, Tuple

class SharedMemoryClient:
    def __init__(self, filename: str = "shared_math.dat", max_size: int = 1024*1024):
        self.filename = filename
        self.max_size = max_size
        
        self.header_format = '64sQ??d'
        self.header_size = struct.calcsize(self.header_format)
        
        if not os.path.exists(self.filename):
            with open(self.filename, "wb") as f:
                f.write(b'\x00' * (self.header_size + max_size))
        
        self.file = open(self.filename, "r+b")
        self.mm = mmap.mmap(self.file.fileno(), self.header_size + max_size)
    
    def call_function(self, func_name: str, *args: Any) -> Tuple[Any, float]:
        args_data = self._pack_arguments(args)
        
        header = struct.pack(
            self.header_format,
            func_name.encode(),
            len(args_data),
            True,   # ready_to_process
            False,  # processing_done
            0.0     # execution_time
        )
        
        self.mm.seek(0)
        self.mm.write(header)
        
        self.mm.seek(self.header_size)
        self.mm.write(args_data)
        
        while True:
            self.mm.seek(0)
            header = self.mm.read(self.header_size)
            _, _, _, done, exec_time = struct.unpack(self.header_format, header)
            
            if done:
                self.mm.seek(self.header_size + len(args_data))
                result = self._unpack_result(func_name)
                return result, exec_time
            
            time.sleep(0.001)
    
    def _pack_arguments(self, args: Tuple[Any, ...]) -> bytes:
        packed = b''
        for arg in args:
            if isinstance(arg, np.ndarray):
                packed += struct.pack('Q', len(arg))
                packed += arg.tobytes()
        return packed
    
    def _unpack_result(self, func_name: str) -> Any:
        if func_name == "add_arrays":
            return struct.unpack('d', self.mm.read(8))[0]
        elif func_name == "multiply_arrays":
            size = struct.unpack('Q', self.mm.read(8))[0]
            return np.frombuffer(self.mm.read(size * 8), dtype=np.float64)
        
        return None
    
    def cleanup(self):
        self.mm.close()
        self.file.close()
