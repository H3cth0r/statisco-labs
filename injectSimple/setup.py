from setuptools import setup, Extension

module = Extension("simpleops", sources=["simpleops.c"])

setup(
    name        ="simpleops",
    version     ="1.0",
    description ="Python C extension for simpleops",
    ext_modules =[module]
)
