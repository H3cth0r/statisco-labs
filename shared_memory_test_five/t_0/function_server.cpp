#include "function_registry.hpp"
#include <csignal>
#include <chrono>
#include <thread>

// Example functions to register
int sumIntegers(int a, int b) {
    return a + b;
}

int* sumArrays(int* arr1, int* arr2, int size) {
    int* result = new int[size];
    for (int i = 0; i < size; i++) {
        result[i] = arr1[i] + arr2[i];
    }
    return result;
}

// Structure for communication
#pragma pack(1)
struct CommandBlock {
    char function_name[64];
    bool request_pending;
    bool response_ready;
    bool should_exit;
};
#pragma pack()

volatile std::sig_atomic_t g_running = true;

void signal_handler(int) {
    g_running = false;
}

int main() {
    signal(SIGINT, signal_handler);
    
    // Create the shared memory for commands
    const char* cmd_filename = "command_block.dat";
    int cmd_fd = open(cmd_filename, O_RDWR | O_CREAT, 0666);
    if (cmd_fd == -1) {
        std::cerr << "Failed to open command file" << std::endl;
        return 1;
    }
    
    if (ftruncate(cmd_fd, sizeof(CommandBlock)) == -1) {
        std::cerr << "Failed to set command file size" << std::endl;
        close(cmd_fd);
        return 1;
    }
    
    CommandBlock* cmd = (CommandBlock*)mmap(
        NULL, sizeof(CommandBlock), PROT_READ | PROT_WRITE, 
        MAP_SHARED, cmd_fd, 0
    );
    
    if (cmd == MAP_FAILED) {
        std::cerr << "Failed to map command memory" << std::endl;
        close(cmd_fd);
        return 1;
    }
    
    // Initialize command block
    cmd->request_pending = false;
    cmd->response_ready = false;
    cmd->should_exit = false;
    
    // Create function registry
    FunctionRegistry registry("functions.dat", 1024 * 1024);
    
    // Register your functions
    registry.registerFunction("addInts", sumIntegers);
    registry.registerFunction("addArrays", sumArrays);
    
    std::cout << "Server started. Waiting for commands..." << std::endl;
    
    while (g_running && !cmd->should_exit) {
        if (cmd->request_pending) {
            try {
                registry.executeFunction(cmd->function_name);
                std::cout << "Executed function: " << cmd->function_name << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Error executing function: " << e.what() << std::endl;
            }
            
            cmd->request_pending = false;
            cmd->response_ready = true;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    // Cleanup
    munmap(cmd, sizeof(CommandBlock));
    close(cmd_fd);
    std::cout << "Server shutdown complete" << std::endl;
    
    return 0;
}
