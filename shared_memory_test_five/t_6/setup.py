from setuptools import setup, Extension
import numpy as np

module = Extension('fast_cpp_caller',
                  sources=['extension.cpp'],
                  include_dirs=[np.get_include()],
                  extra_compile_args=['-std=c++17'])

setup(name='fast_cpp_caller',
      version='1.0',
      ext_modules=[module])
