#include <iostream>
#include "function_registry.hpp"  // Include the previous code here

// Example functions to register
int addInts(int a, int b) {
    return a + b;
}

double addDoubles(double a, double b) {
    return a + b;
}

struct Vector3 {
    double x, y, z;
};

Vector3 addVectors(Vector3 a, Vector3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

int main() {
    try {
        // Create registry with 4KB shared memory
        FunctionRegistry registry("functions.dat", 4096);
        
        // Register functions
        registry.registerFunction("addInts", addInts);
        registry.registerFunction("addDoubles", addDoubles);
        registry.registerFunction("addVectors", addVectors);
        
        std::cout << "Function registry started. Waiting for Python calls...\n";
        
        // Simple event loop to keep the program running
        while (true) {
            // Check for function execution requests
            // You might want to add proper synchronization here
            usleep(1000);  // Sleep for 1ms to prevent busy waiting
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
