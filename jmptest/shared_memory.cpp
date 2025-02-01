#include <sys/mman.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <chrono>

struct SharedData {
    double num1;
    double num2;
    double result;
    bool ready_to_process;
    bool processing_done;
};

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

    printf("C++ program started. Waiting for calculations...\n");

    while (true) {
        if (shared->ready_to_process) {
            double num1 = shared->num1;
            double num2 = shared->num2;

            shared->result = num1 + num2;

            shared->ready_to_process = false;
            shared->processing_done = true;

            printf("Calculated: %.2f + %.2f = %.2f\n", num1, num2, shared->result);
        }

        // usleep(2000);  // 1ms sleep
        usleep(500);  // 1ms sleep
    }

    munmap(shared, sizeof(SharedData));
    close(fd);
    return 0;
}
