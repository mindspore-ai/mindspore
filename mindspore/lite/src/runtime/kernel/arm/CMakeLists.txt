include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)

file(GLOB KERNEL_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/base/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../nnacl/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../nnacl/fp32/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../nnacl/int8/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../nnacl/quantization/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/fp32/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/int8/*.cc
        )
list(REMOVE_ITEM KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../../../../nnacl/opt_op_handler.c)

if (SUPPORT_TRAIN)
file (GLOB TRAIN_KERNEL_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/fp32_grad/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../nnacl/fp32_grad/*.c
        )
endif()

if (PLATFORM_ARM64)
    # assembly
    file(GLOB ASSEMBLY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../../../../nnacl/assembly/arm64/*.s
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../../nnacl/assembly/arm64/*.S)
    set_property(SOURCE ${ASSEMBLY_SRC} PROPERTY LANGUAGE C)
    set(KERNEL_SRC ${KERNEL_SRC} ${ASSEMBLY_SRC})
endif()

#[[
if (PLATFORM_ARM32)
    # assembly
    file(GLOB ASSEMBLY_SRC nnacl/assembly/arm32/*.s
            nnacl/assembly/arm32/*.S
            )
    set_property(SOURCE ${ASSEMBLY_SRC} PROPERTY LANGUAGE C)
    set(KERNEL_SRC ${KERNEL_SRC} ${ASSEMBLY_SRC})
endif()
]]

add_library(cpu_kernel_mid_ OBJECT ${KERNEL_SRC} ${TRAIN_KERNEL_SRC})
