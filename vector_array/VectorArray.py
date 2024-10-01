import ctypes
import mmap
import sys

class VectorArray:
    def __init__(self, size, dtype=ctypes.c_double):
        self.size = size
        self.dtype = dtype
        self.itemsize = ctypes.sizeof(dtype)
        # memory allocation
        self.buffer = mmap.mmap(-1, size*self.itemsize, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, mmap.PROT_READ | mmap.PROT_WRITE)
        # create ctypes array that uses our allocated memory
        self._array = (dtype * size).from_buffer(self.buffer)
        self._array_pointer = None

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self.size)
            return [self._array[i] for i in range(start, stop, step)]
        elif 0 <= index < self.size:
            return self._array[index]
        else:
            raise IndexError("Index out of range")

    def __setitem__(self, index, value):
        if isinstance(index, slice):
            start, stop, step = index.indices(self.size)
            for i, v in zip(range(start, stop, step), value):
                self._array[i] = v
        elif 0 <= index < self.size:
            self._array[index] = value
        else:
            raise IndexError("Index out of range")

    def __len__(self):
        return self.size

    def get_address(self):
        return ctypes.addressof(self._array)

    def get_data_pointer(self):
        if self._array_pointer is None:
            self._array_pointer = ctypes.POINTER(self.dtype)(self._array)
        return self._array_pointer

    def as_array(self):
        return self._array

    def __del__(self):
        if hasattr(self, 'buffer'):
            del self._array
            if self._array_pointer is not None:
                del self._array_pointer
            self.buffer.close()

    @classmethod
    def zeros(cls, size, dtype=ctypes.c_double):
        instance            = cls.__new__(cls)
        instance.size       = size
        instance.dtype      = dtype
        instance.itemsize   = ctypes.sizeof(dtype)
        
        # Directly allocate zeroed memory
        instance.buffer = mmap.mmap(-1, size * instance.itemsize, 
                                    mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, 
                                    mmap.PROT_READ | mmap.PROT_WRITE)
        
        # Create ctypes array from the zeroed buffer
        instance._array = (dtype * size).from_buffer(instance.buffer)
        instance._array_pointer = None
        
        return instance
