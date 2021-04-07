
cmake_minimum_required(VERSION 3.14)
project(benchmark)

if(NOT DEFINED PKG_PATH)
    message(FATAL_ERROR "PKG_PATH not set")
endif()

get_filename_component(PKG_PATH ${PKG_PATH} ABSOLUTE BASE_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(HEADER_PATH ${PKG_PATH}/inference)

option(PLATFORM_ARM64 "build android arm64" OFF)
option(PLATFORM_ARM32 "build android arm32" OFF)

add_compile_definitions(NOT_USE_STL)

if(PLATFORM_ARM64 OR PLATFORM_ARM32)
  add_compile_definitions(ENABLE_NEON)
  add_compile_definitions(ENABLE_ARM)
endif()

if(PLATFORM_ARM64)
  add_compile_definitions(ENABLE_ARM64)
endif()

if(PLATFORM_ARM32)
  add_compile_definitions(ENABLE_ARM32)
  add_definitions(-mfloat-abi=softfp -mfpu=neon)
endif()

set(CMAKE_C_FLAGS "${CMAKE_ENABLE_C99} ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    message(STATUS "build benchmark with debug info")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DDebug -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDebug -g")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=default")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=default")
else()
    message(STATUS "build benchmark release version")
    set(CMAKE_C_FLAGS "-fPIC -fPIE -D_FORTIFY_SOURCE=2 -O3 -Wall -Werror -fstack-protector-strong -Wno-attributes \
    -Wno-deprecated-declarations -Wno-missing-braces ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-fPIC -fPIE -D_FORTIFY_SOURCE=2 -O3 -Wall -Werror -fstack-protector-strong -Wno-attributes \
    -Wno-deprecated-declarations -Wno-missing-braces -Wno-overloaded-virtual ${CMAKE_CXX_FLAGS}")
    string(REPLACE "-g" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    string(REPLACE "-g" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

add_subdirectory(src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${HEADER_PATH})
set(SRC_FILES
        benchmark/benchmark.cc
        benchmark/load_input.c
)
add_executable(benchmark ${SRC_FILES})
target_link_libraries(benchmark net -lm -pthread)
