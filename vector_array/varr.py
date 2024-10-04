import ctypes
import mmap

def create_vector_array(size, dtype=ctypes.c_double):
    itemsize = ctypes.sizeof(dtype)
    buffer = mmap.mmap(-1, size * itemsize, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, mmap.PROT_READ | mmap.PROT_WRITE)
    array = (dtype * size).from_buffer(buffer)

    def __getitem__(index):
        if isinstance(index, slice):
            start, stop, step = index.indices(size)
            return [array[i] for i in range(start, stop, step)]
        elif 0 <= index < size:
            return array[index]
        else:
            raise IndexError("Index out of range")

    def __setitem__(index, value):
        if isinstance(index, slice):
            start, stop, step = index.indices(size)
            for i, v in zip(range(start, stop, step), value):
                array[i] = v
        elif 0 <= index < size:
            array[index] = value
        else:
            raise IndexError("Index out of range")

    def __len__():
        return size

    def get_address():
        return ctypes.addressof(array)

    def get_data_pointer():
        return ctypes.POINTER(dtype)(array)

    def as_array():
        return array

    def cleanup():
        nonlocal array, buffer
        del array
        buffer.close()

    return {
        '__getitem__': __getitem__,
        '__setitem__': __setitem__,
        '__len__': __len__,
        'get_address': get_address,
        'get_data_pointer': get_data_pointer,
        'as_array': as_array,
        'cleanup': cleanup,
    }

def zeros(size, dtype=ctypes.c_double): 
    return create_vector_array(size, dtype)

# Example usage:
# arr = zeros(10)
# print(arr['__getitem__'](slice(None)))  # Get all elements
# arr['__setitem__'](0, 5.0)
# print(arr['__getitem__'](0))
# arr['cleanup']()
