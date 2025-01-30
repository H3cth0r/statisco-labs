# Interpreter Hacking

## References
[Inside the Python Virtual Machine](https://leanpub.com/insidethepythonvirtualmachine/read)


The thing is that I want to implement some kind of functionality in a C extension, to be able to load the functionalities from another ".so" file. For example, imagine I have created a purely C library of functionalities and I want to be able to load or inject them to be accessible from python, by using the helper C extension. The thing is that writing the C exttensions is a very long process and requires to add a lot of code. I dont want the need of using methods to have the return type as "PyObject" and the python arguments; I want to have some kind of C-extension that enabled loading a C library with normal functions. Is there any way i can have my functionality library just like this:
// mylib.c
#include "func_metadata.h" int add(int a, int b) { return a + b; } and the wrapper C python exntesion does the whole process to be able to load the file in order to access this functionality from python?
