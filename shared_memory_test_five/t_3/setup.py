from setuptools import setup, Extension
import numpy as np

module = Extension(
    'fast_functions',
    sources=[
        'function_registry.cpp',
        'pymodule.cpp',
        'my_library.cpp'
    ],
    include_dirs=[np.get_include()],
    extra_compile_args=['-std=c++17'],
    language='c++'
)

setup(
    name='fast_functions',
    version='1.0',
    ext_modules=[module],
    install_requires=['numpy']
)
