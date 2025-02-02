#include <sys/mman.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <chrono>

#define MAX_ARRAY_SIZE 1000

// Ensure proper alignment and packing
#pragma pack(1)
struct SharedData {
    int32_t array_size;
    bool ready_to_process;
    bool processing_done;
    double array1[MAX_ARRAY_SIZE];
    double array2[MAX_ARRAY_SIZE];
    double result[MAX_ARRAY_SIZE];
};
#pragma pack()

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

    // Initialize flags
    shared->ready_to_process = false;
    shared->processing_done = false;

    while (true) {
        if (shared->ready_to_process) {
            printf("Got new data to process, size: %d\n", shared->array_size);
            
            int size = shared->array_size;
            
            // Perform array addition
            for(int i = 0; i < size; i++) {
                shared->result[i] = shared->array1[i] + shared->array2[i];
            }

            // Mark processing as done
            shared->processing_done = true;
            shared->ready_to_process = false;

            printf("Processed arrays of size %d\n", size);
            printf("First few elements: %.2f + %.2f = %.2f\n", 
                   shared->array1[0], shared->array2[0], shared->result[0]);
        }
        usleep(500);
    }

    munmap(shared, sizeof(SharedData));
    close(fd);
    return 0;
}
