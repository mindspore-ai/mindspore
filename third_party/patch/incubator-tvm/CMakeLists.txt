cmake_minimum_required(VERSION 3.2)
project(tvm C CXX)
set(TVM_DIR ${CMAKE_CURRENT_SOURCE_DIR})
# Utility functions
include(${TVM_DIR}/cmake/util/Util.cmake)
include(${TVM_DIR}/cmake/util/FindCUDA.cmake)

# include directories
include_directories(AFTER "${TVM_DIR}/include")
include_directories(AFTER "${TVM_DIR}/src")
include_directories(AFTER "${TVM_DIR}")
include_directories(AFTER "${TVM_DIR}/src/schedule")

include_directories(AFTER "${TVM_DIR}/3rdparty/dmlc-core/include")
include_directories(AFTER "${TVM_DIR}/3rdparty/dlpack/include")
include_directories(AFTER "${TVM_DIR}/3rdparty/compiler-rt")
include_directories(AFTER "${TVM_DIR}/3rdparty/rang/include")

# lib contain dlopen and dlclose
set(TVM_RUNTIME_LINKER_LIBS ${CMAKE_DL_LIBS})

# add source group
file(GLOB_RECURSE GROUP_SOURCE "${TVM_DIR}/src/*.cc" "src/*.cc")
file(GLOB_RECURSE GROUP_INCLUDE "${TVM_DIR}/src/*.h"
        "${TVM_DIR}/include/*.h" "src/*.h" "include/*.h")
assign_source_group("Source" ${GROUP_SOURCE})
assign_source_group("Include" ${GROUP_INCLUDE})

file(GLOB COMPILER_SRCS
        "pre_activate/gpu/*.cc"
        ${TVM_DIR}/src/api/*.cc
        ${TVM_DIR}/src/arithmetic/*.cc
        ${TVM_DIR}/src/autotvm/*.cc
        ${TVM_DIR}/src/codegen/*.cc
        ${TVM_DIR}/src/lang/*.cc
        ${TVM_DIR}/src/pass/*.cc
        ${TVM_DIR}/src/op/*.cc
        ${TVM_DIR}/src/node/*.cc
        ${TVM_DIR}/src/schedule/*.cc
        ${TVM_DIR}/src/runtime/*.cc
        ${TVM_DIR}/src/runtime/vm/*.cc
        ${TVM_DIR}/src/runtime/vm/profiler/*.cc
        ${TVM_DIR}/src/codegen/stackvm/*.cc)

file(GLOB_RECURSE RELAY_SRCS ${TVM_DIR}/src/relay/*.cc)
list(APPEND COMPILER_SRCS ${RELAY_SRCS})

file(GLOB DATATYPE_SRCS ${TVM_DIR}/src/codegen/datatype/*.cc)
list(APPEND COMPILER_SRCS ${DATATYPE_SRCS})

file(GLOB COMPILER_VERILOG_SRCS ${TVM_DIR}/src/codegen/verilog/*.cc)
list(APPEND COMPILER_SRCS ${COMPILER_VERILOG_SRCS})

file(GLOB TOPI_SRCS ${TVM_DIR}/topi/src/*.cc)

file(GLOB RUNTIME_SRCS
        ${TVM_DIR}/src/runtime/*.cc
        ${TVM_DIR}/src/runtime/vm/*.cc
        ${TVM_DIR}/src/runtime/stub/*.cc
        ${TVM_DIR}/src/runtime/stackvm/*.cc)


file(GLOB COMPILER_OFF_SRCS
        ${TVM_DIR}/src/codegen/opt/build_*_off.cc)

list(REMOVE_ITEM COMPILER_OFF_SRCS
        ${TVM_DIR}/src/codegen/opt/build_cuda_off.cc)
set(USE_CUDA "ON")
list(APPEND COMPILER_SRCS ${COMPILER_OFF_SRCS})
# Module rules
include(${TVM_DIR}/cmake/modules/CUDA.cmake)

set(CMAKE_C_FLAGS_AKG  -pipe -Wall -fPIC -fstack-protector-all)
set(CMAKE_C_FLAGS_AKG ${CMAKE_C_FLAGS_AKG} -Wl,-z,relro,-z,now,-z,noexecstack)

set(CMAKE_CXX_FLAGS_AKG  -std=c++11 -pipe -Wall -fPIC -fstack-protector-all)
set(CMAKE_CXX_FLAGS_AKG ${CMAKE_CXX_FLAGS_AKG} -Wl,-z,relro,-z,now,-z,noexecstack)

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    message("-- Build in Debug mode")
    set(CMAKE_C_FLAGS_AKG ${CMAKE_C_FLAGS_AKG} -O0 -g -rdynamic)
    set(CMAKE_CXX_FLAGS_AKG ${CMAKE_CXX_FLAGS_AKG} -O0 -g -rdynamic)
else()
    message("-- Build in Release mode")
    set(CMAKE_C_FLAGS_AKG ${CMAKE_C_FLAGS_AKG} -O2 -Werror)
    set(CMAKE_CXX_FLAGS_AKG ${CMAKE_CXX_FLAGS_AKG} -O2 -Werror)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_CXX_COMPILER_VERSION
        VERSION_GREATER 7.0)
    set(CMAKE_CXX_FLAGS_AKG ${CMAKE_CXX_FLAGS_AKG} -faligned-new)
endif()

add_library(tvm SHARED ${COMPILER_SRCS} ${RUNTIME_SRCS} ${TOPI_SRCS})

target_link_libraries(tvm ${TVM_LINKER_LIBS} ${TVM_RUNTIME_LINKER_LIBS})
target_compile_options(tvm PRIVATE
        $<$<COMPILE_LANGUAGE:C>:${CMAKE_C_FLAGS_AKG}>
        $<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CXX_FLAGS_AKG}>)
target_include_directories(tvm PRIVATE "${TVM_DIR}/topi/include")
install(TARGETS tvm)