from multiprocessing import shared_memory
import time

SHARED_MEM_NAME = "c_program"
SHARED_MEM_SIZE = 256

class SharedMemoryClient:
    def __init__(self):
        # Connect to shared memory
        self.shm = shared_memory.SharedMemory(name=SHARED_MEM_NAME)
        self.shared_mem = self.shm.buf

    def send_request(self, operation, num1, num2):
        # Write data to shared memory
        request_data = f"{operation} {num1} {num2}".encode('utf-8')
        self.shared_mem[1:1+len(request_data)] = request_data
        self.shared_mem[0] = ord('1')  # Set flag to "1" (data ready)

        # Wait for the result
        while self.shared_mem[0] != ord('2'):
            time.sleep(0.01)  # Avoid busy-waiting

        # Read the result
        result = bytes(self.shared_mem[1:SHARED_MEM_SIZE]).split(b'\x00', 1)[0].decode('utf-8')
        self.shared_mem[0] = ord('0')  # Reset flag
        return result

    def close(self):
        self.shm.close()


if __name__ == "__main__":
    client = SharedMemoryClient()

    print("Addition result:", client.send_request("add", 5, 3))
    print("Subtraction result:", client.send_request("sub", 10, 4))
    print("Multiplication result:", client.send_request("mul", 7, 6))

    client.close()
