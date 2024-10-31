import subprocess

# Compile memalloc.c
result1 = subprocess.run(
    ["gcc", "-c", "memalloc.c", "-o", "memalloc.o"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)
output1 = result1.stdout.decode()
error1 = result1.stderr.decode()

if result1.returncode != 0:
    print("Compilation of memalloc.c failed.")
    print("Error message:", error1)
    exit(1)

# Compile main.c
result2 = subprocess.run(
    ["gcc", "-c", "main.c", "-o", "main.o"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)
output2 = result2.stdout.decode()
error2 = result2.stderr.decode()

if result2.returncode != 0:
    print("Compilation of main.c failed.")
    print("Error message:", error2)
    exit(1)

# Link object files
result3 = subprocess.run(
    ["gcc", "memalloc.o", "main.o", "-o", "main", "-pthread"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)
output3 = result3.stdout.decode()
error3 = result3.stderr.decode()

if result3.returncode == 0:
    print("Compilation and linking succeeded.")
    print(output3)
else:
    print("Linking failed.")
    print("Error message:", error3)
    print("Output message:", output3)
