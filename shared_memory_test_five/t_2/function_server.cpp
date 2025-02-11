#include "function_registry.hpp"
#include <csignal>
#include <chrono>
#include <thread>

int addInts(int a, int b) {
    return a + b;
}

int addMoreInts(int a, int b, int c) {
    return a + b + c;
}

FixedArray addArrays(const FixedArray& a, const FixedArray& b) {
    FixedArray result;
    result.size = std::min(a.size, b.size);
    for (size_t i = 0; i < result.size; i++) {
        result.data[i] = a.data[i] + b.data[i];
    }
    return result;
}

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
    
    cmd->request_pending = false;
    cmd->response_ready = false;
    cmd->should_exit = false;
    
    FunctionRegistry registry("functions.dat", 1024 * 1024);
    registry.registerFunction("addInts", addInts);
    registry.registerFunction("addArrays", addArrays);
    registry.registerFunction("addMoreInts", addMoreInts);
    
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
        
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    
    munmap(cmd, sizeof(CommandBlock));
    close(cmd_fd);
    std::cout << "Server shutdown complete" << std::endl;
    
    return 0;
}
