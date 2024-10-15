# VectorArray V1

This is an attempt to create an optimized vector/array data structure, optimized
purely from python, by using ctypes and mmap. Right now I've tried three different alternatives
to test the creation of a **zeros** array. 

### Load from the creation of a Vector Array class
This implementation has the zeros method in. It is kinda fast, but there is a bottle neck that doesnt 
enable to reduce the time. The thing is that the python class instantiation is pretty slow. By using
slots you are able to reduce the time about a 20%, but still its not possible to make it even faster
than numpy. Even thought, it is a little bit faster than the tinygrads Tensor instantiation, but maybe 
this could increase when adding more functionalities.

### Load in Numpy
Numpy is fast, but C extensions would do event better.

### Create vector on a single method/function
This actually makes it very fast, it takes half the time numpy takes to load it; the thing is that it
is pretty difficult to add more functionalities to array, this way.

### Load from dict
This one is still even faster than numpy and lets you store functionalities in a dictionary, but you most 
invoke all the backbone functions by yourself.

### Improved Load from dict
This one is an improved version of load from dict, but enablind to have mor functionalities, but still, 
this is not very natural and efficient.

## Results
```
load_by_instance_VectorArray took 0.000059 seconds to execute.
load_single_numpy_array took 0.000015 seconds to execute.
load_single_vector_method took 0.000009 seconds to execute.
load_from_dict took 0.000006 seconds to execute.
using_more_complex_dict took 0.000011 seconds to execute.
```
