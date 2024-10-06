from setuptools import setup, Extension

module = Extension("vectorarray", sources=["vectorarray.c"])

setup(
    name        ="vectorarray",
    version     ="1.0",
    description ="Python C extension for vector arrays",
    ext_modules =[module]
)
