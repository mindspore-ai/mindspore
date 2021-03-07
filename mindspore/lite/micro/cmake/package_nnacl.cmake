include_directories(${LITE_DIR})
set(NNACL_DIR ${LITE_DIR}/nnacl)
file(GLOB KERNEL_SRC
    ${NNACL_DIR}/*.c
    ${NNACL_DIR}/base/*.c
    ${NNACL_DIR}/fp32/*.c
    ${NNACL_DIR}/int8/*.c
)

if(MICRO_BUILD_ARM64)
    file(GLOB ASSEMBLY_SRC ${NNACL_DIR}/assembly/arm64/*.S)
    set_property(SOURCE ${ASSEMBLY_SRC} PROPERTY LANGUAGE C)
endif()

if(MICRO_BUILD_ARM32A)
    file(GLOB ASSEMBLY_SRC ${NNACL_DIR}/assembly/arm32/*.S)
    set_property(SOURCE ${ASSEMBLY_SRC} PROPERTY LANGUAGE C)
endif()

set(NNACL_OPS ${KERNEL_SRC} ${ASSEMBLY_SRC})
