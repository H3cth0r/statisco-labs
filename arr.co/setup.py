import subprocess
import sysconfig
import re

# Python Include
python_include  = sysconfig.get_paths()['include']
python_libs     = python_include.replace('include', 'lib')

result = subprocess.run(
    ["gcc", "-g", "-shared", "-O", "arrco.c", "-o", "arrco.o"],
    # ["gcc", "-I", python_include, "-L", python_libs, "-g", "-shared", "-O", "arrco.c", "-o", "arrco.o"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

output = result.stdout.decode()
error = result.stderr.decode()

if result.returncode == 0:
    print("Compilation succeeded.")
    print(output)
else:
    print("Compilation failed.")
    print("Error message:", error)
    print("Output message:", output)

objdump_result = subprocess.run(
    ["objdump", "-M", "intel", "-d", "arrco.o"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

objdump_output = objdump_result.stdout

method_addresses = {}
methods_to_find = ['init_array', 'set_element', 'get_element', 'set_all_elements', 'to_python_list']

for method in methods_to_find:
    match = re.search(rf'^([0-9a-f]+) <{method}>:', objdump_output, re.MULTILINE)
    if match:
        address = match.group(1).lstrip('0')
        method_addresses[method] = f"0x{address}"

text_start_match = re.search(r'^([0-9a-f]+) <([^>]+)>:', objdump_output.split('.text:')[1], re.MULTILINE)
if text_start_match:
    text_start = f"0x{text_start_match.group(1).lstrip('0')}"

subprocess.call(["objcopy", "--only-section=.text", "-O", "binary", "arrco.o", "text.o"])

with open("text.o", "rb") as fil:
    with open("byteArrCo.py", "w") as ut:
        ut.write("code = ")
        ut.write(str(fil.read()))
        ut.write("\n\n")
        ut.write(f"text_start = {text_start}\n")
        for method, address in method_addresses.items():
            ut.write(f"{method}_address = {address}\n")

print("Done setup")
