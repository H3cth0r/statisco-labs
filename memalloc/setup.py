import subprocess

result = subprocess.run(
        ["gcc", "-c", "memalloc.c", "-o", "memalloc.o"],
        ["gcc", "-c", "main.c", "-o", "main.o"],
        ["gcc", "memalloc.o", "main.o", "-o", "main", "-pthread"],
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
