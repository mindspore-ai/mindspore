cmake_minimum_required(VERSION 3.14)
project (Lite_Internal)
set(TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)

include_directories(${TOP_DIR})

file(GLOB KERNEL_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/../nnacl/arithmetic_common.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../nnacl/fp32/activation.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../nnacl/fp32/arithmetic_self.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../nnacl/fp32/arithmetic.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../nnacl/fp32/matmul.c
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
        ${CMAKE_CURRENT_SOURCE_DIR}/src/lite_session.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ms_tensor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/common/string.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/common/vector.cc
        ${TOP_DIR}/src/common/log_adapter.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../core/gvar/logging_level.cc
        ${TOP_DIR}/src/runtime/allocator.cc
        )

if (PLATFORM_ARM64)
    # assembly
    file(GLOB ASSEMBLY_SRC
            ${CMAKE_CURRENT_SOURCE_DIR}/../nnacl/assembly/arm64/MatmulFp32OptRemain.S
            ${CMAKE_CURRENT_SOURCE_DIR}/../nnacl/assembly/arm64/MatmulFp32Opt.S)
    set_property(SOURCE ${ASSEMBLY_SRC} PROPERTY LANGUAGE C)
    set(KERNEL_SRC ${KERNEL_SRC} ${ASSEMBLY_SRC})
    add_library(mslite_internal SHARED ${CCSRC} ${KERNEL_SRC} ${TRAIN_KERNEL_SRC})
    target_link_libraries(mslite_internal log)
endif()

