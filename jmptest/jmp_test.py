import ctypes

# Load the compiled shared library
lib = ctypes.CDLL("./jmp_hack.so")

# Function prototypes
lib.hack_add.argtypes = [ctypes.c_int, ctypes.c_int]
lib.hack_add.restype = ctypes.c_int

lib.modify_and_jump.argtypes = [ctypes.c_int]
lib.modify_and_jump.restype = None

print("Calling hack_add(5, 3)...")
original_result = lib.hack_add(5, 3)
print(f"Original result: {original_result}")

print("Modifying the result and jumping back...")
lib.modify_and_jump(999)  # Change the return value to 999

# hack_add() will "return" 999 instantly
final_result = lib.hack_add(5, 3)
print(f"Final result: {final_result}")
