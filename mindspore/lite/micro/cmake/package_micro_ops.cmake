include_directories(${NNACL_DIR}/..)

set(CMSIS_SRC ${NNACL_DIR}/../micro/build/cmsis)
if(MICRO_CMSIS_X86)
    message("*****build cmsis x86 codes****")
    include_directories(${CMSIS_SRC}/CMSIS/Core/Include)
    include_directories(${CMSIS_SRC}/CMSIS/DSP/Include)
    include_directories(${CMSIS_SRC}/CMSIS/NN/Include)
    file(GLOB RUNTIME_KERNEL_CMSIS_SRC
            ${CMSIS_SRC}/CMSIS/NN/Source/BasicMathFunctions/*.c
            ${CMSIS_SRC}/CMSIS/NN/Source/ActivationFunctions/*.c
            ${CMSIS_SRC}/CMSIS/NN/Source/ConcatenationFunctions/*.c
            ${CMSIS_SRC}/CMSIS/NN/Source/ConvolutionFunctions/*.c
            ${CMSIS_SRC}/CMSIS/NN/Source/FullyConnectedFunctions/*.c
            ${CMSIS_SRC}/CMSIS/NN/Source/NNSupportFunctions/*.c
            ${CMSIS_SRC}/CMSIS/NN/Source/PoolingFunctions/*.c
            ${CMSIS_SRC}/CMSIS/NN/Source/ReshapeFunctions/*.c
            ${CMSIS_SRC}/CMSIS/NN/Source/SoftmaxFunctions/*.c
            )
endif()

########################### files ###########################
file(GLOB RUNTIME_KERNEL_SRC
        ${NNACL_DIR}/kernel/fp32/*.c
        ${NNACL_DIR}/kernel/int8/*.c
        )
if(MICRO_CMSIS_X86)
    set(RUNTIME_OPS ${RUNTIME_KERNEL_SRC} ${RUNTIME_TRAIN_SRC} ${RUNTIME_KERNEL_CMSIS_SRC})
else()
    set(RUNTIME_OPS ${RUNTIME_KERNEL_SRC} ${RUNTIME_TRAIN_SRC})
endif()

