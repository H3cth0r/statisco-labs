
```
project_root/
├── CMakeLists.txt
├── include/
│   ├── functions.hpp
│   └── shared_memory.hpp
├── src/
│   ├── functions.cpp
│   └── main.cpp
└── python/
    └── shared_memory_client.py
```


```
client = SharedMemoryClient()

result, exec_time = client.call_function("add_arrays", arr1, arr2, len(arr1))
print(f"Sum: {result}, Execution time: {exec_time}ms")

result, exec_time = client.call_function("multiply_arrays", arr1, arr2, len(arr1))
print(f"Multiplication result: {result}, Execution time: {exec_time}ms")

client.cleanup()
```
