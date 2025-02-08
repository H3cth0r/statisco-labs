cmake_minimum_required(VERSION 3.10)
project(DynamicFunctionLib CXX)

# Set C++ standard and compilation flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# Add debug symbols in Debug mode
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

# Optimization in Release mode
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Create the shared library for functions
add_library(functions SHARED
    src/functions.cpp
)

# Create the main executable
add_executable(function_server
    src/main.cpp
)

# Add include directories
target_include_directories(functions PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(function_server PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link the functions library to the server
target_link_libraries(function_server
    functions
)

# Installation rules
install(TARGETS functions function_server
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)

install(FILES 
    include/FunctionRegistry.hpp
    include/shared_memory.hpp
    DESTINATION include
)
