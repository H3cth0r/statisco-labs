# Analysis

## core components
- shared memory: allows different processes to access the same regions of memory. This avoids overhead of copying data between processes. Changes made by one process are inmediately visible to others.
- function registry: server maintains a registry of functions that can execute. These functions are made available to python client.
- command block: small suze structre in shared memory that acts as a control channel. Client uses it to tell server which function to run and when it's done.
- function metadata: information about each registered function(name, number of arguments, argument types, return types) is stored in shared memory. Allows client and server to agree on how to pass data.
- mmap: The mmap system call us the key to using shared memory. Maps a file into the process's address space.

## Structures
- `#pragma once`: common header guard, ensuring file is included only one
- `MAX_ARRAY_SIZE`: defines the max size of the fixed-size arrays for communication.
- `FixedArray`: structure to represent fixed-size arrays. Contains an array of integers(data) and the actual number of elements to be used. TODO will this work for floats?.
- `#pragma pack(1)`: crucial for shared memory. Tells compiles to pack the structure members tightly, without any padding bytes between them. Ensures that the layout of the structure is the same in both cpp and python code. Without this, offsets of members might be different due to compiler specific padding rules.

### FunctionMetadata: describes registered functions
- name: function's name
- `num_args`: number of arguments
- `return_offset`: offset in shared memory where the return value will be stored.
- `return_size`: size in bytes of the return value. 
- `arg_offsets`: array of offsets where each argument is stored in shared memory. 
- `arg_sizes`: array of sizes in bytes of each argument. 

### SharedHeader: describes overall layout of the shared memory region
- `num_functions`: num of registered functions.
- `data_start_offset`: offset to the start ofn the data area(where argumetns and return values are stored).
- `functions`: an array of FunctionMetadata structed, one for each registered function.


### SharedMemoryManager
Manages the shared memory segment. Handles opening, mapping and allocating space within the shared memory.
- `filename`: name of the file used to back the shared memory.
- `fd`: file descriptior for the shared memory file.
- `mapped_memory`: pointer to the start of the mapped memory region. This is how the program accesses the shared memory.
- `total_size`: total size of the shared memory region.
- `header`: pointer to the SharedMemoryManager at the beginning of the shared memory.
- `data_section`: pointer to the start of the data area within the shared memory.
- Constructor:
    - Opens or creates the shared memory file (open).
    - Sets the file size (ftruncate). This is important because the file must be large enough to hold all the data.
    - Maps the file into memory (mmap). `PROT_READ | PROT_WRITE` allows both reading and writing. `MAP_SHARED` makes changes visible to other processes.
    - Initializes the header and `data_section` pointers.
    - Initializes the memory allocated for the arguments and return values of the functions to 0.

- `allocateBlock<T>()`: Calculates the next available offset within the data section of the shared memory. It does this by finding the maximum offset used by any existing function's arguments or return value. This is a simple (but potentially wasteful) allocation strategy.
- `getPointer<T>(offset)`: Returns a typed pointer (e.g., `int*, FixedArray*`) to a specific offset within the shared memory. This is how the code accesses data at known offsets.
- `**getHeader(): ** Returns` the SharedMemoryHeader.
- Destructor: Unmaps the shared memory (munmap) and closes the file descriptor (close). This is essential to release resources.

## `FunctionWrapper` and `TypedFunctionWrapper` classes

#### FunctionWrapper
Abstract base class for function wrappers. It defines the interface that all function wrappers must implement:
- `execute(SharedMemoryManager& memory)`: the method that will be called to actually execute the function.

#### TypedFunctionWrapper
Template class that adapts a specific `C++` function(with its know return type and argument types) to the FucntionWrapper interface.
- func: a `std::function` object that holds the actual cpp function to be called. `std::function` is a versatile way to store any callable object(function pointer, lambda, etc.).
- metadata: Pointer to the function metadata.
- `get_args(SharedMemoryManager& memory, std::index_sequence<I...>)`: uses a parameter pack and `std::index_sequence` to create a `std::tuple` containing the arguments, fetched from their respective offsets in shared memory. This is a compile-time mechanism to unpack the arguments.
- Constructor:
    - Stores the provided function (f) in the func member.
    - Increments the function counter in the shared memory header.
    - Copies the function's name into the metadata.
    - Calculates and stores the offsets and sizes of the arguments in the metadata. This uses a fold expression ((...), ...) to iterate over the argument types.
    - Calculates and stores the offset and size of the return value (if there is one).

- execute(SharedMemoryManager& memory): The core of the function execution:
    - Calls `get_args` to retrieve the arguments from shared memory as a tuple.
    - Uses std::apply to call the stored function (func) with the unpacked arguments. std::apply takes a callable object and a tuple, and calls the callable with the tuple elements as arguments.
    - If the function has a return value (not void), it stores the result at the designated return_offset in shared memory.

### FunctionRegistry Class
- Purpose: Manages the registration and execution of functions.
- memory: A SharedMemoryManager instance, handling the underlying shared memory.
- functions: An `unordered_map` that stores function wrappers. The key is the function name (a string), and the value is a `unique_ptr` to a FunctionWrapper object. Using `unique_ptr` ensures proper memory management of the wrappers.
- Constructor: Initializes the SharedMemoryManager.
- registerFunction(const std::string& name, Ret(*func)(Args...)): Registers a function.
    - Creates a TypedFunctionWrapper for the given function.
    - Stores the wrapper in the functions map.
- executeFunction(const std::string& name): Executes a registered function.
    - Looks up the function wrapper by name in the functions map.
    - If found, calls the wrapper's execute method, passing the SharedMemoryManager instance.
    - If not found, throws an exception.

## Sever
- Includes: Includes the header file where the classes are defined, and headers for signal handling, time, and threading.
- addInts, addArrays These functions will be stored inside the registry.
- CommandBlock: (Same as FunctionMetadata, crucial #pragma pack(1)). Defines the structure for communication commands from the client to the server:
    - `function_name`: The name of the function to execute.
    - `request_pending`: Set to true by the client when it wants the server to execute a function.
    - `response_ready`: Set to true by the server when it has finished executing the function.
    - `should_exit`: Set to true by the client to request the server termination.
- `g_running` and `signal_handler`: Handle graceful shutdown on Ctrl+C (SIGINT).
- main:
    1. Setup:
        - Sets up the signal handler for SIGINT.
        - Opens/creates/maps the `command_block.dat` file (similar to the shared memory for functions, but this one is for commands).
        - Initializes the CommandBlock in shared memory.
        - Creates a FunctionRegistry, registering the addInts and addArrays functions.
    2. Main Loop:
        - Continuously checks the `request_pending` flag in the CommandBlock.
        - If `request_pending` is true:
            - Calls registry.executeFunction to execute the requested function.
            - Prints a message indicating which function was executed.
            - Sets `request_pending` to false and `response_ready` to true.
        - Sleeps for a short period (10 milliseconds) to avoid busy-waiting.
        - Checks the `should_exit` flag for a clean server termination.
    3. Cleanup: Unmaps and closes the command block shared memory.

## Python Code Breakdown
```
class FixedArray(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("data", ctypes.c_int * 1024),
        ("size", ctypes.c_size_t)
    ]

class CommandBlock(ctypes.Structure):
   # ...

class FunctionMetadata(ctypes.Structure):
    # ...

class SharedMemoryHeader(ctypes.Structure):
    # ...
```
- _pack_ = 1: The equivalent of #pragma pack(1) in C++. Ensures the same structure layout in Python as in C++.
- _fields_: Defines the fields of each structure, mirroring the C++ definitions. ctypes.c_int, ctypes.c_size_t, ctypes.c_char are C-compatible data types.
- FixedArray, CommandBlock, FunctionMetadata, SharedMemoryHeader: Python versions of the C++ structures.

### FunctionClient class
- `__init__`:
    - Opens the `command_block.dat` and functions.dat files in read/write binary mode ("r+b").
    - Creates memory maps (mmap.mmap) for both files. The 0 size means to map the entire file.
    - Creates ctypes structure instances (CommandBlock, SharedMemoryHeader) from the memory map buffers. This allows you to access the shared memory as if it were these structures.
    - Builds a dictionary (self.functions) mapping function names to FunctionMetadata objects. This is used to look up function information.
- `execute_function(self, name: str, *args: Any) -> Any`:
    - Argument Validation: Checks if the function name exists and if the number of arguments is correct.
    - Write Arguments: Iterates through the provided arguments (*args):
        - Calculates the offset and size for each argument using the metadata.
        - Converts and writes the argument to the shared memory:
            - Integers and Floats: Converted to bytes using .to_bytes().
            - Lists, Tuples, NumPy Arrays: Converted to a FixedArray structure and then written as bytes.
            - Other Types: Raises an error (unsupported type).
    - Send Command:
        - Sets the function_name in the CommandBlock.
        - Sets request_pending to True.
        - Flushes the memory map (self.cmd_mm.flush()) to ensure the changes are written to the shared memory.
    - Wait for Response: Waits in a loop until request_pending becomes False. Includes a short sleep to avoid busy-waiting.
    - Read Result:
        - If response_ready is True:
            - Reads the return value from the shared memory based on metadata.return_offset and metadata.return_size.
            - Converts the result back to the appropriate Python type (integer or list).
        - Returns the result (or None if there's no return value).
- close(self):
    - Sets cmd.should_exit for a clean server termination.
    - Closes the file handles. This is important to release resources.
