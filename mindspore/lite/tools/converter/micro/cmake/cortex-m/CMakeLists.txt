cmake_minimum_required(VERSION 3.12)
project(Micro)

string(REPLACE "-Werror" "" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
string(REPLACE "-Werror" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

set(CMAKE_VERBOSE_MAKEFILE on)
set(MICRO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)

include_directories(${NNACL_DIR}/..)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-attributes -fdata-sections -ffunction-sections")

file(GLOB KERNEL_SRC
        ${NNACL_DIR}/*.c
        ${NNACL_DIR}/fp32/*.c
        ${NNACL_DIR}/infer/*.c
        ${NNACL_DIR}/base/*.c
        ${NNACL_DIR}/fp32_grad/*.c
        ${NNACL_DIR}/kernel/*.c
        ${NNACL_DIR}/experimental/*.c
        ${NNACL_DIR}/int8/*.c
        ${NNACL_DIR}/fp32_sparse/*.c
        ${NNACL_DIR}/infer/string/*.c
        ${NNACL_DIR}/infer/control/*.c
        )

list(REMOVE_ITEM KERNEL_SRC ${NNACL_DIR}/infer/shape_fusion_infer.c)

string(REPLACE "-fvisibility=hidden" "-fvisibility=default" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")

add_library(nnacl_static STATIC ${KERNEL_SRC})
target_compile_options(nnacl_static PRIVATE -fPIC)

target_link_options(nnacl_static PRIVATE -Wl,-z,relro,-z,now,-z,noexecstack)

include(${TOP_DIR}/cmake/utils.cmake)
include(${TOP_DIR}/cmake/external_libs/cmsis.cmake)

set(CMSIS_DIR ${CMAKE_BINARY_DIR}/cmsis)
message("build cmsis kernels")
include_directories(${CMSIS_DIR}/CMSIS/Core/Include)
include_directories(${CMSIS_DIR}/CMSIS/DSP/Include)
include_directories(${CMSIS_DIR}/CMSIS/NN/Include)

file(GLOB CMSIS_OPS
        ${CMSIS_DIR}/CMSIS/NN/Source/BasicMathFunctions/*.c
        ${CMSIS_DIR}/CMSIS/NN/Source/ActivationFunctions/*.c
        ${CMSIS_DIR}/CMSIS/NN/Source/ConcatenationFunctions/*.c
        ${CMSIS_DIR}/CMSIS/NN/Source/ConvolutionFunctions/*.c
        ${CMSIS_DIR}/CMSIS/NN/Source/FullyConnectedFunctions/*.c
        ${CMSIS_DIR}/CMSIS/NN/Source/NNSupportFunctions/*.c
        ${CMSIS_DIR}/CMSIS/NN/Source/PoolingFunctions/*.c
        ${CMSIS_DIR}/CMSIS/NN/Source/ReshapeFunctions/*.c
        ${CMSIS_DIR}/CMSIS/NN/Source/SoftmaxFunctions/*.c
        )
list(REMOVE_ITEM CMSIS_OPS ${CMSIS_DIR}/CMSIS/NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.c)
add_library(cmsis_nn STATIC ${CMSIS_OPS})

include_directories(${MICRO_DIR}/coder/)
set(WRAPPER_DIR ${MICRO_DIR}/coder/wrapper/)
file(GLOB WRAPPER_SRC
        ${WRAPPER_DIR}/base/*.c
        ${WRAPPER_DIR}/fp32/*.c
        ${WRAPPER_DIR}/int8/*.c
        )

# generate static library
add_library(wrapper STATIC ${WRAPPER_SRC})
