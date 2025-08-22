cmake_minimum_required(VERSION 3.16)
project(FastEyeBLAST_CPP)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(nlohmann_json QUIET)

# Check for local nlohmann/json
set(LOCAL_JSON_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/include/nlohmann/json.hpp")

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /W4")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zi /Od")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /DNDEBUG")
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Source files
set(SOURCES
    Fast_EyeBLAST_IVT.cpp
)

# Create executable
add_executable(fast_eyeblast ${SOURCES})

# Handle JSON library
if(nlohmann_json_FOUND)
    message(STATUS "Using system nlohmann/json")
    target_link_libraries(fast_eyeblast nlohmann_json::nlohmann_json)
elseif(EXISTS ${LOCAL_JSON_HEADER})
    message(STATUS "Using local nlohmann/json from include/nlohmann/json.hpp")
    target_include_directories(fast_eyeblast PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
else()
    message(FATAL_ERROR "nlohmann/json not found. Please install it or download json.hpp to include/nlohmann/json.hpp")
endif()

# Threading support
find_package(Threads REQUIRED)
target_link_libraries(fast_eyeblast Threads::Threads)

# Set output directory
set_target_properties(fast_eyeblast PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Installation rules
install(TARGETS fast_eyeblast
    RUNTIME DESTINATION bin
)

# Copy example data if it exists
if(EXISTS "${CMAKE_SOURCE_DIR}/GazeBaseVR_IVT")
    install(DIRECTORY "${CMAKE_SOURCE_DIR}/GazeBaseVR_IVT"
        DESTINATION share/fast_eyeblast
        FILES_MATCHING PATTERN "*.json")
endif()

# Print build information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}") 