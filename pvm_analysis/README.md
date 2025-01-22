

```
typedef struct _is {

    struct _is *next;
    struct _ts *tstate_head;

    PyObject *modules;
    PyObject *modules_by_index;
    PyObject *sysdict;
    PyObject *builtins;
    PyObject *importlib;

    PyObject *codec_search_path;
    PyObject *codec_search_cache;
    PyObject *codec_error_registry;
    int codecs_initialized;
    int fscodec_initialized;

    PyObject *builtins_copy;
} PyInterpreterState;
```

1. `*next`reference to another interpreter instance as multiple python interpreters can exist within the same process.
2. `*tstate_head` field points to the main thread of execution. If the program is multithreaded, then the interpreter is shared by all threads created by the program.
3. The `modules`, `modules_by_index`, `sysdict`, `builtins` and `importlib` are self-explanatory. They are defined as instances of `PyObject` which is the root type of all python objects in the virtual Machine world. 
4. The `codec*` related fields hold information that helps with the location and loading of encodings. These are very important for decoding bytes.

## Core Process
- Python is a compiled language that converts source code into bytecode
- The bytecode is executed by Python's virtual machine
- Python uses a stack-based execution model for operations

## Compilation Steps
1. Source code is parsed into a parse tree
2. Parse tree is transformed into an Abstract Syntax Tree (AST)
3. Symbol table is generated
4. AST is converted into a control flow graph
5. Control flow graph becomes a code object

## Code Execution
- Code objects contain bytecode instructions
- Python's virtual machine executes these instructions
- Instructions manipulate values on a stack
- Common operations include LOAD_FAST, BINARY_MULTIPLY, RETURN_VALUE

## Tokenization
- Source code is broken into tokens (identifiers, operators, literals, etc.)
- Indentation is important - generates special INDENT/DEDENT tokens
- Multiple physical lines can form one logical line
- Tokenization is the first step in the parsing process

## References
[Inside the Python Virtual Machine](https://leanpub.com/insidethepythonvirtualmachine/read)
