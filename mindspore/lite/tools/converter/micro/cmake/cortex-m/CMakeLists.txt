cmake_minimum_required(VERSION 3.12)
project(Micro)
set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -fstack-protector-strong -Wno-strict-aliasing")

set(CMAKE_VERBOSE_MAKEFILE on)
set(MICRO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)
include_directories(${NNACL_DIR}/..)

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
