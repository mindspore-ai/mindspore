set(CMSIS_DIR ${CMAKE_BINARY_DIR}/cmsis)
message("build cmsis kernels")
include_directories(${CMSIS_DIR}/CMSIS/Core/Include)
include_directories(${CMSIS_DIR}/CMSIS/DSP/Include)
include_directories(${CMSIS_DIR}/CMSIS/NN/Include)

file(REMOVE ${CMSIS_DIR}/CMSIS/NN/Source/NNSupportFunctions/arm_q7_to_q15_reordered_no_shift.c)

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

