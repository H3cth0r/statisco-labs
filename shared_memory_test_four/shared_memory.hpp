#pragma once
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <string>
#include "FunctionRegistry.hpp"

struct SharedMemoryHeader {
    char function_name[64];
    size_t data_size;
    bool ready_to_process;
    bool processing_done;
    double execution_time_ms;
};

class SharedMemoryManager {
    private:
        std::string filename_;
        size_t total_size_;
        int fd_;
        char* data_;
        SharedMemoryHeader* header_;
        char* payload_;

    public:
      SharedMemoryManager(const std::string& filename, size_t max_size) 
          : filename_(filename), total_size_(sizeof(SharedMemoryHeader) + max_size) {
          
          fd_ = open(filename.c_str(), O_RDWR | O_CREAT, 0666);
          if (fd_ == -1) throw std::runtime_error("Failed to open shared memory file");
          
          if (ftruncate(fd_, total_size_) == -1) {
              close(fd_);
              throw std::runtime_error("Failed to set file size");
          }
          
          data_ = static_cast<char*>(mmap(
              NULL, total_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0
          ));
          
          if (data_ == MAP_FAILED) {
              close(fd_);
              throw std::runtime_error("Failed to map memory");
          }
          
          header_ = reinterpret_cast<SharedMemoryHeader*>(data_);
          payload_ = data_ + sizeof(SharedMemoryHeader);
      }
      
      ~SharedMemoryManager() {
          if (data_ != MAP_FAILED) {
              munmap(data_, total_size_);
          }
          if (fd_ != -1) {
              close(fd_);
          }
      }
      
      void process_requests(FunctionRegistry& registry) {
          while (true) {
              if (header_->ready_to_process && !header_->processing_done) {
                  const FunctionMetadata* metadata = registry.get_metadata(header_->function_name);
                  if (metadata) {
                      auto start_time = std::chrono::high_resolution_clock::now();
                      
                      // Execute function
                      char* result_buffer = payload_ + metadata->total_size;
                      registry.execute(header_->function_name, payload_, result_buffer);
                      
                      auto end_time = std::chrono::high_resolution_clock::now();
                      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                          end_time - start_time);
                      
                      header_->execution_time_ms = duration.count() / 1000.0;
                      header_->processing_done = true;
                      header_->ready_to_process = false;
                  }
              }
              usleep(500);
          }
      }
}
