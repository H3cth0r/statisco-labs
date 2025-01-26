from setuptools import setup, Extension

module = Extension(
    "interpstate",
    sources=["interpstate.c"],
    include_dirs=[]  # No need for Python internal headers
)

setup(
    name="interpstate",
    version="1.0",
    description="A module to inspect loaded Python modules",
    ext_modules=[module],
)
