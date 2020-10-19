cmake_minimum_required(VERSION 3.14)
project (Lite_Internal)
set(TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)
set(CMAKE_CXX_COMPILER ${CMAKE_C_COMPILER})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions")
include_directories(${TOP_DIR})
add_compile_definitions(ENABLE_NNACL_INFER_SHAPE)

file(GLOB KERNEL_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/../nnacl/arithmetic_common.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../nnacl/fp32/activation.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../nnacl/fp32/arithmetic_self.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../nnacl/fp32/arithmetic.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../nnacl/fp32/matmul.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../nnacl/fp32/reduce.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../nnacl/fp32/arithmetic.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/fp32/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/common/*.cc
        )
if (SUPPORT_TRAIN)
    file(GLOB TRAIN_KERNEL_SRC
            ${KERNEL_SRC}
            ${CMAKE_CURRENT_SOURCE_DIR}/../nnacl/fp32_grad/activation_grad.c
            ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/fp32_grad/*.cc
            )
endif ()

list(REMOVE_ITEM KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../nnacl/opt_op_handler.c)

set(CCSRC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/common/vector.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/common/string.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/lite_session.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/allocator.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ms_tensor.cc
        )

if (PLATFORM_ARM64)
    # assembly
    file(GLOB ASSEMBLY_SRC
            ${CMAKE_CURRENT_SOURCE_DIR}/../nnacl/assembly/arm64/MatmulFp32Opt.S
            ${CMAKE_CURRENT_SOURCE_DIR}/../nnacl/assembly/arm64/MatVecMulFp32.S
            ${CMAKE_CURRENT_SOURCE_DIR}/../nnacl/assembly/arm64/MatmulFp32.S)
    set_property(SOURCE ${ASSEMBLY_SRC} PROPERTY LANGUAGE C)
    set(KERNEL_SRC ${KERNEL_SRC} ${ASSEMBLY_SRC})
    add_library(mslite_internal SHARED ${CCSRC} ${KERNEL_SRC} ${TRAIN_KERNEL_SRC})
endif()
