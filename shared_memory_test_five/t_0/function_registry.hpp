#pragma once
#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <functional>
#include <memory>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>  // Added for close() and ftruncate()
#include <any>
#include <typeindex>

// Forward declarations
class FunctionWrapper;
class SharedMemoryManager;

// Class to manage shared memory allocation and access
class SharedMemoryManager {
private:
    struct MemoryBlock {
        size_t offset;
        size_t size;
        std::type_index type;
    };
    
    std::string filename;
    int fd;
    void* mapped_memory;
    size_t total_size;
    std::vector<MemoryBlock> blocks;
    size_t current_offset;

public:
    SharedMemoryManager(const std::string& fname, size_t size) 
        : filename(fname), total_size(size), current_offset(0) {
        fd = open(filename.c_str(), O_RDWR | O_CREAT, 0666);
        if (fd == -1) {
            throw std::runtime_error("Failed to open shared memory file");
        }
        
        if (ftruncate(fd, size) == -1) {
            close(fd);
            throw std::runtime_error("Failed to set file size");
        }
        
        mapped_memory = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (mapped_memory == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("Failed to map memory");
        }
    }
    
    template<typename T>
    size_t allocateBlock(size_t count = 1) {
        size_t block_size = sizeof(T) * count;
        if (current_offset + block_size > total_size) {
            throw std::runtime_error("Not enough shared memory");
        }
        
        size_t allocated_offset = current_offset;
        blocks.push_back({allocated_offset, block_size, std::type_index(typeid(T))});
        current_offset += block_size;
        
        return allocated_offset;
    }
    
    template<typename T>
    T* getPointer(size_t offset) {
        return reinterpret_cast<T*>(static_cast<char*>(mapped_memory) + offset);
    }
    
    ~SharedMemoryManager() {
        if (mapped_memory != MAP_FAILED) {
            munmap(mapped_memory, total_size);
        }
        if (fd != -1) {
            close(fd);
        }
    }
};

// Base class for function wrappers
class FunctionWrapper {
public:
    virtual void execute(SharedMemoryManager& memory) = 0;
    virtual ~FunctionWrapper() = default;
};

// Template class for specific function types
template<typename Ret, typename... Args>
class TypedFunctionWrapper : public FunctionWrapper {
private:
    std::function<Ret(Args...)> func;
    std::vector<size_t> arg_offsets;
    size_t return_offset;

    template<std::size_t... I>
    std::tuple<Args...> get_args(SharedMemoryManager& memory, std::index_sequence<I...>) {
        return std::tuple<Args...>{(*memory.getPointer<std::remove_reference_t<Args>>(arg_offsets[I]))...};
    }

public:
    TypedFunctionWrapper(std::function<Ret(Args...)> f, SharedMemoryManager& memory) 
        : func(f) {
        // Allocate memory blocks for arguments
        (arg_offsets.push_back(memory.allocateBlock<std::remove_reference_t<Args>>()), ...);
        
        // Allocate memory block for return value if not void
        if constexpr (!std::is_void_v<Ret>) {
            return_offset = memory.allocateBlock<Ret>();
        }
    }

    void execute(SharedMemoryManager& memory) override {
        auto args = get_args(memory, std::index_sequence_for<Args...>{});
        
        if constexpr (std::is_void_v<Ret>) {
            std::apply(func, args);
        } else {
            Ret result = std::apply(func, args);
            *memory.getPointer<Ret>(return_offset) = result;
        }
    }
};

// Main registry class
class FunctionRegistry {
private:
    SharedMemoryManager memory;
    std::unordered_map<std::string, std::unique_ptr<FunctionWrapper>> functions;

public:
    FunctionRegistry(const std::string& filename, size_t size) 
        : memory(filename, size) {}

    template<typename Ret, typename... Args>
    void registerFunction(const std::string& name, Ret(*func)(Args...)) {
        auto wrapper = std::make_unique<TypedFunctionWrapper<Ret, Args...>>(
            std::function<Ret(Args...)>(func), memory);
        functions[name] = std::move(wrapper);
    }

    void executeFunction(const std::string& name) {
        auto it = functions.find(name);
        if (it == functions.end()) {
            throw std::runtime_error("Function not found: " + name);
        }
        it->second->execute(memory);
    }
};
