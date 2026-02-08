cmake_minimum_required(VERSION 3.10)
project(msprobe_debug_stub LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(msprobe_debug_stub SHARED msprobe_debug_stub.cc)

set_target_properties(msprobe_debug_stub PROPERTIES
    OUTPUT_NAME "msprobe_debug_stub"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
