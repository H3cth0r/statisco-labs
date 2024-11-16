import subprocess
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Start the timer
        result = func(*args, **kwargs)  # Call the function
        end_time = time.time()  # End the timer
        duration = end_time - start_time  # Calculate duration
        print(f"Function '{func.__name__}' took {duration:.8f} seconds to complete.")
        return result
    return wrapper

@timer
def communicate_with_c(operation, num1, num2):
    process = subprocess.Popen(
            ['./c_program'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True
    )
    input_data = f"{operation} {num1} {num2}\n"
    process.stdin.write(input_data)
    process.stdin.flush()

    result = process.stdout.readline().strip()

    process.stdin.close()
    process.stdout.close()
    return result 

if __name__ == "__main__":
    print("Addition result:", communicate_with_c("add", 5, 3))
    print("Subtraction result:", communicate_with_c("sub", 10, 4))
    print("Multiplication result:", communicate_with_c("mul", 7, 6))
