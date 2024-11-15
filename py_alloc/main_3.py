from setuptools import setup, Extension

malloc_allocator_module = Extension(
    'malloc_allocator',
    sources=['malloc_allocator.c'],
)

setup(
    name='malloc_allocator',
    version='1.0',
    description='MallocAllocator extension module',
    ext_modules=[malloc_allocator_module],
)
