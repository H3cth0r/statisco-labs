# Injecting C VectorArray to Python

## Steps
1. Compile by running: 
```
python setup.py build_ext --inplace
```
2. Use objdump dissasembler to check the executable in assembly form.
```
cd build/temp

objdump -M intel -d vectorarray.o
```
3. Copy the text part from the `vectorarray.o` to another `.o` file:
```
objcopy --only-section=.text -O binary vectorarray.o text.o
```
4. Use the `extractor.py` to copy the text section to a python bytes file called
`text.py`
