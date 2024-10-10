import subprocess

path_o = "./build/temp.linux-x86_64-cpython-310/vectorops.o"

# call(["python", "setup.py", "build_ext", "--inplace"])
# call(["cp", path_o, "./"])
# call(["objcopy", "--only-section=.text", "-O", "binary", "vectorops.o", "text.o"])

# gcc -g -c vectorops.c -o vectorops.o
result = subprocess.run(
    # ["gcc", "-g", "-c", "vectorops.c", "-o", "vectorops.o"],
    ["gcc", "-O2", "vectorops.c", "-o", "vectorops.o"],
    stdout=subprocess.PIPE,  # Capture standard output
    stderr=subprocess.PIPE   # Capture standard error
)

# Get the output and error messages
output = result.stdout.decode()  # Decode the output
error = result.stderr.decode()    # Decode the error message

# Print the output and error messages
if result.returncode == 0:
    print("Compilation succeeded.")
    print(output)
else:
    print("Compilation failed.")
    print("Error message:", error)

print("Output message:", output)

subprocess.call(["objcopy", "--only-section=.text", "-O", "binary", "vectorops.o", "text.o"])


with open("text.o", "rb") as fil:
    with open("text.py", "w") as ut:
        ut.write("code=")
        ut.write(str(fil.read()))
print("Done setup")
