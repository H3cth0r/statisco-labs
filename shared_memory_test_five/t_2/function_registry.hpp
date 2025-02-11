#pragma once
#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <functional>
#include <memory>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <array>

constexpr size_t MAX_ARRAY_SIZE = 1024;

struct FixedArray {
    int data[MAX_ARRAY_SIZE];
    size_t size;
};

#pragma pack(1)
struct FunctionMetadata {
    char name[64];
    size_t num_args;
    size_t return_offset;
    size_t return_size;
    size_t arg_offsets[10];
    size_t arg_sizes[10];
};

struct SharedMemoryHeader {
    size_t num_functions;
    size_t data_start_offset;
    FunctionMetadata functions[100];
};
#pragma pack()

class SharedMemoryManager {
private:
    std::string filename;
    int fd;
    void* mapped_memory;
    size_t total_size;
    SharedMemoryHeader* header;
    char* data_section;

public:
    SharedMemoryManager(const std::string& fname, size_t size) 
        : filename(fname), total_size(size) {
        fd = open(filename.c_str(), O_RDWR | O_CREAT, 0666);
        if (fd == -1) throw std::runtime_error("Failed to open shared memory file");
        
        if (ftruncate(fd, size) == -1) {
            close(fd);
            throw std::runtime_error("Failed to set file size");
        }
        
        mapped_memory = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (mapped_memory == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("Failed to map memory");
        }

        header = static_cast<SharedMemoryHeader*>(mapped_memory);
        header->num_functions = 0;
        header->data_start_offset = sizeof(SharedMemoryHeader);
        data_section = static_cast<char*>(mapped_memory) + header->data_start_offset;

        // init memory
        std::memset(data_section, 0, size - sizeof(SharedMemoryHeader));
    }

    template<typename T>
    size_t allocateBlock() {
        size_t offset = header->data_start_offset;
        for (size_t i = 0; i < header->num_functions; i++) {
            for (size_t j = 0; j < header->functions[i].num_args; j++) {
                offset = std::max(offset, header->functions[i].arg_offsets[j] + header->functions[i].arg_sizes[j]);
            }
            if (header->functions[i].return_size > 0) {
                offset = std::max(offset, header->functions[i].return_offset + header->functions[i].return_size);
            }
        }
        return offset;
    }

    template<typename T>
    T* getPointer(size_t offset) {
        return reinterpret_cast<T*>(static_cast<char*>(mapped_memory) + offset);
    }

    SharedMemoryHeader* getHeader() { return header; }

    ~SharedMemoryManager() {
        if (mapped_memory != MAP_FAILED) munmap(mapped_memory, total_size);
        if (fd != -1) close(fd);
    }
};

class FunctionWrapper {
public:
    virtual void execute(SharedMemoryManager& memory) = 0;
    virtual ~FunctionWrapper() = default;
};

template<typename Ret, typename... Args>
class TypedFunctionWrapper : public FunctionWrapper {
private:
    std::function<Ret(Args...)> func;
    FunctionMetadata* metadata;

    template<std::size_t... I>
    std::tuple<Args...> get_args(SharedMemoryManager& memory, std::index_sequence<I...>) {
        return std::tuple<Args...>{(*memory.getPointer<std::remove_reference_t<Args>>(metadata->arg_offsets[I]))...};
    }

public:
    TypedFunctionWrapper(std::function<Ret(Args...)> f, SharedMemoryManager& memory, const std::string& name) 
        : func(f) {
        metadata = &memory.getHeader()->functions[memory.getHeader()->num_functions++];
        std::strncpy(metadata->name, name.c_str(), 63);
        metadata->name[63] = '\0';
        metadata->num_args = sizeof...(Args);

        size_t arg_idx = 0;
        ((metadata->arg_offsets[arg_idx] = memory.allocateBlock<std::remove_reference_t<Args>>(),
          metadata->arg_sizes[arg_idx] = sizeof(std::remove_reference_t<Args>),
          arg_idx++), ...);

        if constexpr (!std::is_void_v<Ret>) {
            metadata->return_offset = memory.allocateBlock<Ret>();
            metadata->return_size = sizeof(Ret);
        } else {
            metadata->return_size = 0;
        }
    }

    void execute(SharedMemoryManager& memory) override {
        auto args = get_args(memory, std::index_sequence_for<Args...>{});
        if constexpr (std::is_void_v<Ret>) {
            std::apply(func, args);
        } else {
            Ret result = std::apply(func, args);
            *memory.getPointer<Ret>(metadata->return_offset) = result;
        }
    }
};

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
            std::function<Ret(Args...)>(func), memory, name);
        functions[name] = std::move(wrapper);
    }

    void executeFunction(const std::string& name) {
        auto it = functions.find(name);
        if (it == functions.end()) throw std::runtime_error("Function not found: " + name);
        it->second->execute(memory);
    }
};
