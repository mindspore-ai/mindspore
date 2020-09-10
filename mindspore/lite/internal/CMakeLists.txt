cmake_minimum_required(VERSION 3.14)
project (Lite_Internal)
set(TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)

include_directories(${TOP_DIR})

file(GLOB_RECURSE C_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
file(GLOB KERNEL_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/../nnacl/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../nnacl/fp32/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../nnacl/int8/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../nnacl/quantization/*.c
        )
list(REMOVE_ITEM KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../nnacl/opt_op_handler.c)

set(CCSRC
        ${TOP_DIR}/src/common/log_adapter.cc
        ${TOP_DIR}/src/runtime/allocator.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../core/gvar/logging_level.cc
        )

if (PLATFORM_ARM64)
    # assembly
    file(GLOB ASSEMBLY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../nnacl/assembly/arm64/*.s
            ${CMAKE_CURRENT_SOURCE_DIR}/../nnacl/assembly/arm64/*.S)
    set_property(SOURCE ${ASSEMBLY_SRC} PROPERTY LANGUAGE C)
    set(KERNEL_SRC ${KERNEL_SRC} ${ASSEMBLY_SRC})
    add_library(mslite_internal SHARED ${C_SRC} ${CCSRC} ${KERNEL_SRC})
    target_link_libraries(mslite_internal log)
endif()

