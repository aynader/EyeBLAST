cmake_minimum_required(VERSION 3.5)
project(PP_EyeBLAST LANGUAGES CXX)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Performance flags (match emp-tool build)
add_compile_options(-O3 -march=native -maes -mrdseed)

# Find emp-tool (core library)
find_package(emp-tool REQUIRED)

# Find OpenSSL for crypto
find_package(OpenSSL REQUIRED)

# Find Threads library
find_package(Threads REQUIRED)

# ---- MOVED UP: define your vendor include-dir variables _before_ you use them ----
# Header-only modules (vendor paths)
set(EMP_OT_INCLUDE_DIR     "${CMAKE_SOURCE_DIR}/third_party/emp-ot")
set(EMP_SH2PC_INCLUDE_DIR  "${CMAKE_SOURCE_DIR}/third_party/emp-sh2pc/emp-sh2pc")

# <<< ADDED: build emp-sh2pc so its library target exists
add_subdirectory(third_party/emp-sh2pc)

# <<< ADDED: now pull in our headers and vendored headers
include_directories(
  ${CMAKE_SOURCE_DIR}/include       # your public headers
  ${EMP_OT_INCLUDE_DIR}              # emp-ot headers, if you ever need them
  ${EMP_SH2PC_INCLUDE_DIR}           # emp-sh2pc headers (for setup_semi_honest, etc)
)

# <<< ADDED: build the FastEyeBLAST core as a static library
add_library(fasteyeblast STATIC
  src/fast_eyeblast.cpp
)
target_include_directories(fasteyeblast PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

# <<< ADDED: pp_coarse executable
add_executable(pp_coarse
  src/pp_coarse.cpp
)
target_link_libraries(pp_coarse PRIVATE
  fasteyeblast
  ${EMP-TOOL_LIBRARIES}    # from find_package(emp-tool)
  emp-sh2pc                # now links the library built by add_subdirectory()
  Threads::Threads
)

# <<< ADDED: pp_refine executable
add_executable(pp_refine
  src/pp_refine.cpp
)
target_link_libraries(pp_refine PRIVATE
  fasteyeblast
  ${EMP-TOOL_LIBRARIES}
  emp-sh2pc
  Threads::Threads
)

# <<< ADDED: pp_coordinate protocol coordination executable
add_executable(pp_coordinate
  src/pp_coordinate.cpp
)
target_link_libraries(pp_coordinate PRIVATE
  fasteyeblast
  ${EMP-TOOL_LIBRARIES}
  emp-sh2pc
  Threads::Threads
)

# <<< ADDED: pp_main orchestration executable
add_executable(pp_main
  src/pp_main.cpp
)
target_link_libraries(pp_main PRIVATE
  fasteyeblast
  ${EMP-TOOL_LIBRARIES}
  emp-sh2pc
  Threads::Threads
)

# ---- your existing test target (unchanged) ----

add_executable(test_emp-tool
  tests/test_emp-tool.cpp
)

target_include_directories(test_emp-tool PRIVATE
  ${EMP-TOOL_INCLUDE_DIRS}
  ${EMP_OT_INCLUDE_DIR}
  ${EMP_SH2PC_INCLUDE_DIR}
)

target_link_libraries(test_emp-tool PRIVATE
  ${EMP-TOOL_LIBRARIES}
  OpenSSL::Crypto
  Threads::Threads
)
add_executable(compare_plain_vs_gc
  src/compare_plain_vs_gc.cpp
)
target_link_libraries(compare_plain_vs_gc PRIVATE
  fasteyeblast            # already defined static lib
  Threads::Threads        # not strictly needed but harmless
)