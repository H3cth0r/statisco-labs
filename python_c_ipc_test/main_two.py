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

class CPersistentProcess:
    def __init__(self, c_program_path):
        # Start the C program as a subprocess
        self.process = subprocess.Popen(
            [c_program_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True
        )

    @timer
    def send_request(self, operation, num1, num2):
        # Send the operation and numbers to the C program
        input_data = f"{operation} {num1} {num2}\n"
        self.process.stdin.write(input_data)
        self.process.stdin.flush()

        # Read the result from the C program
        result = self.process.stdout.readline().strip()
        return result

    def close(self):
        # Close the subprocess
        self.process.stdin.close()
        self.process.terminate()


# Example usage
if __name__ == "__main__":
    c_program = "./c_program_loop"  # Path to your compiled C program
    process = CPersistentProcess(c_program)

    print("Addition result:", process.send_request("add", 5, 3))
    print("Subtraction result:", process.send_request("sub", 10, 4))
    print("Multiplication result:", process.send_request("mul", 7, 6))

    # Close the process when done
    process.close()
