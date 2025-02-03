#include <sys/mman.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <chrono>

#define MAX_ARRAY_SIZE 1000

enum Operation {
    ADD = 0,
    SUBTRACT = 1,
    MULTIPLY = 2
};

#pragma pack(1)
struct SharedData {
    int32_t array_size;
    int32_t operation;  // Added operation field
    bool ready_to_process;
    bool processing_done;
    double array1[MAX_ARRAY_SIZE];
    double array2[MAX_ARRAY_SIZE];
    double result[MAX_ARRAY_SIZE];
    double execution_time_ms;  // Added timing field
};
#pragma pack()

void add_arrays(double* arr1, double* arr2, double* result, int size) {
    for(int i = 0; i < size; i++) {
        result[i] = arr1[i] + arr2[i];
    }
}

void subtract_arrays(double* arr1, double* arr2, double* result, int size) {
    for(int i = 0; i < size; i++) {
        result[i] = arr1[i] - arr2[i];
    }
}

void multiply_arrays(double* arr1, double* arr2, double* result, int size) {
    for(int i = 0; i < size; i++) {
        result[i] = arr1[i] * arr2[i];
    }
}

int main() {
    const char* filename = "shared_math.dat";
    
    int fd = open(filename, O_RDWR | O_CREAT, 0666);
    if (fd == -1) {
        perror("Error opening/creating file");
        return 1;
    }

    if (ftruncate(fd, sizeof(SharedData)) == -1) {
        perror("Error setting file size");
        close(fd);
        return 1;
    }

    SharedData* shared = (SharedData*)mmap(
        NULL, 
        sizeof(SharedData), 
        PROT_READ | PROT_WRITE, 
        MAP_SHARED, 
        fd, 
        0
    );

    if (shared == MAP_FAILED) {
        perror("Error mapping file");
        close(fd);
        return 1;
    }

    printf("C++ program started. Waiting for arrays to process...\n");
    printf("Shared memory size: %lu bytes\n", sizeof(SharedData));

    shared->ready_to_process = false;
    shared->processing_done = false;
    shared->execution_time_ms = 0.0;

    while (true) {
        if (shared->ready_to_process) {
            int size = shared->array_size;
            Operation op = static_cast<Operation>(shared->operation);
            
            auto start_time = std::chrono::high_resolution_clock::now();
            
            switch(op) {
                case ADD:
                    add_arrays(shared->array1, shared->array2, shared->result, size);
                    printf("Performing addition\n");
                    break;
                case SUBTRACT:
                    subtract_arrays(shared->array1, shared->array2, shared->result, size);
                    printf("Performing subtraction\n");
                    break;
                case MULTIPLY:
                    multiply_arrays(shared->array1, shared->array2, shared->result, size);
                    printf("Performing multiplication\n");
                    break;
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            shared->execution_time_ms = duration.count() / 1000.0;  // Convert to milliseconds

            shared->processing_done = true;
            shared->ready_to_process = false;

            printf("Processed arrays of size %d\n", size);
            printf("Operation completed in %.3f ms\n", shared->execution_time_ms);
            printf("First few elements: %.2f op %.2f = %.2f\n", 
                   shared->array1[0], shared->array2[0], shared->result[0]);
        }
        usleep(500);
    }

    munmap(shared, sizeof(SharedData));
    close(fd);
    return 0;
}
