file(GLOB KERNEL_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/base/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/opclib/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/opclib/fp32/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/opclib/int8/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/opclib/quantization/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/fp32/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/int8/*.cc
        )

if (PLATFORM_ARM64)
    # assembly
    file(GLOB ASSEMBLY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/opclib/assembly/arm64/*.s
            ${CMAKE_CURRENT_SOURCE_DIR}/opclib/assembly/arm64/*.S)
    set_property(SOURCE ${ASSEMBLY_SRC} PROPERTY LANGUAGE C)
    set(KERNEL_SRC ${KERNEL_SRC} ${ASSEMBLY_SRC})
endif()

if (PLATFORM_ARM32)
    # assembly
    file(GLOB ASSEMBLY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/opclib/assembly/arm32/*.s
            ${CMAKE_CURRENT_SOURCE_DIR}/opclib/assembly/arm32/*.S
            )
    set_property(SOURCE ${ASSEMBLY_SRC} PROPERTY LANGUAGE C)
    set(KERNEL_SRC ${KERNEL_SRC} ${ASSEMBLY_SRC})
endif()

add_library(cpu_kernel_mid_ OBJECT ${KERNEL_SRC})
add_subdirectory(opclib)
