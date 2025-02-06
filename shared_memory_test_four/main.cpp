#include "shared_memory.hpp"
#include "functions.hpp"
#include <iostream>
#include <signal.h>

volatile bool running = true;

void signal_handler(int) {
    running = false;
}

int main() {
    try {
        signal(SIGINT, signal_handler);
        signal(SIGTERM, signal_handler);

        FunctionRegistry registry;
        register_functions(registry);

        SharedMemoryManager shm("shared_math.dat", 1024*1024);  // 1MB buffer
        
        std::cout << "Server started. Processing requests..." << std::endl;
        
        while (running) {
            shm.process_requests(registry);
        }
        
        std::cout << "\nShutting down gracefully..." << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
