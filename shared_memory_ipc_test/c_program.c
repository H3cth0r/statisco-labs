#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#define SHARED_MEM_NAME "/c_program"
#define SHARED_MEM_SIZE 256

int main() {
    int shm_fd = shm_open(SHARED_MEM_NAME, O_RDWR, 0666);
    if (shm_fd == -1) {
        perror("shm_open");
        return 1;
    }

    char *shared_mem = mmap(0, SHARED_MEM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shared_mem == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    while (1) {
        if (shared_mem[0] == '1') {
            char operation[10];
            int num1, num2, result = 0;

            sscanf(shared_mem + 1, "%s %d %d", operation, &num1, &num2);

            if (strcmp(operation, "add") == 0) {
                result = num1 + num2;
            } else if (strcmp(operation, "sub") == 0) {
                result = num1 - num2;
            } else if (strcmp(operation, "mul") == 0) {
                result = num1 * num2;
            } else {
                snprintf(shared_mem + 1, SHARED_MEM_SIZE - 1, "Error: Unkwown operation");
                shared_mem[0] = '2';
                continue;
            }
            snprintf(shared_mem + 1, SHARED_MEM_SIZE - 1, "Result: %d", result);
            shared_mem[0] = '2';

        }
        usleep(1000);
    }
    munmap(shared_mem, SHARED_MEM_SIZE);
    close(shm_fd);

    return 0;
}
