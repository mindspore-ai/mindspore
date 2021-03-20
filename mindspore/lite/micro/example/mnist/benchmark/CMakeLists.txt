
cmake_minimum_required(VERSION 3.14)
project(benchmark)

if(NOT DEFINED MODEL_LIB)
    message(FATAL_ERROR "MODEL_LIB not set")
endif()

get_filename_component(MODEL_LIB ${MODEL_LIB} ABSOLUTE BASE_DIR ${CMAKE_CURRENT_BINARY_DIR})

function(parse_lib_info lib_full_path lib_name lib_path)
    string(FIND "${lib_full_path}" "/" POS REVERSE)
    math(EXPR POS "${POS} + 1")
    string(SUBSTRING ${lib_full_path} 0 ${POS} path)
    set(${lib_path} ${path} PARENT_SCOPE)
    string(SUBSTRING ${lib_full_path} "${POS}" "-1" name)
    set(${lib_name} ${name} PARENT_SCOPE)
endfunction(parse_lib_info)

parse_lib_info(${MODEL_LIB} MODEL_LIB_NAME MODEL_LIB_PATH)

message("project name: ${MODEL_LIB_NAME}")

option(MICRO_BUILD_ARM64 "build android arm64" OFF)
option(MICRO_BUILD_ARM32A "build android arm32" OFF)

if(MICRO_BUILD_ARM64 OR MICRO_BUILD_ARM32A)
  add_compile_definitions(ENABLE_NEON)
  add_compile_definitions(ENABLE_ARM)
endif()

if(MICRO_BUILD_ARM64)
  add_compile_definitions(ENABLE_ARM64)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8.2-a+dotprod")
endif()

if(MICRO_BUILD_ARM32A)
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
    set(CMAKE_C_FLAGS "-fPIC -fPIE -D_FORTIFY_SOURCE=2 -O2 -Wall -Werror -fstack-protector-strong -Wno-attributes \
    -Wno-deprecated-declarations -Wno-missing-braces ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-fPIC -fPIE -D_FORTIFY_SOURCE=2 -O2 -Wall -Werror -fstack-protector-strong -Wno-attributes \
    -Wno-deprecated-declarations -Wno-missing-braces -Wno-overloaded-virtual ${CMAKE_CXX_FLAGS}")
endif()
link_directories(${MODEL_LIB_PATH})
include(benchmark.cmake)
add_executable(benchmark ${SRC_FILES})
target_link_libraries(benchmark ${MODEL_LIB_NAME} -lm -pthread)

