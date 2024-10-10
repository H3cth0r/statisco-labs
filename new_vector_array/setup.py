
from setuptools import setup, Extension

module = Extension("vectorops", sources=["vectorops.c"])

setup(
    name        ="vectorops",
    version     ="1.0",
    description ="Python C extension for vectorops",
    ext_modules =[module]
)
