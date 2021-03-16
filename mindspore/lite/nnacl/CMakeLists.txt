project(nnacl)

set(NNACL_DIR ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(NNACL_DIR)

if(PLATFORM_ARM32 OR PLATFORM_ARM64)
    if("${CMAKE_BUILD_TYPE}" STREQUAL "Release" AND DEFINED ARCHS)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -fstrict-aliasing \
        -ffunction-sections -fdata-sections -ffast-math -Wno-shorten-64-to-32")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -fstrict-aliasing \
        -ffunction-sections -fdata-sections -ffast-math -Wno-shorten-64-to-32")
    endif()
    if("${CMAKE_BUILD_TYPE}" STREQUAL "Release" AND NOT DEFINED ARCHS)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -fomit-frame-pointer -fstrict-aliasing \
        -ffunction-sections -fdata-sections -ffast-math")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -fomit-frame-pointer -fstrict-aliasing \
        -ffunction-sections -fdata-sections -ffast-math")
    endif()
endif()
if("${X86_64_SIMD}" STREQUAL "avx")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -msse4.1 -mavx -mavx2")
    set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -msse4.1 -mavx -mavx2")
endif()

########################### files ###########################
file(GLOB KERNEL_SRC
    ${NNACL_DIR}/*.c
    ${NNACL_DIR}/fp32/*.c
    ${NNACL_DIR}/int8/*.c
    ${NNACL_DIR}/infer/*.c
    ${NNACL_DIR}/base/*.c
)

if(SUPPORT_TRAIN)
    file(GLOB TRAIN_SRC ${NNACL_DIR}/fp32_grad/*.c)
endif()

if(PLATFORM_ARM64)
    file(GLOB ASSEMBLY_SRC ${NNACL_DIR}/assembly/arm64/*.S)
    set_property(SOURCE ${ASSEMBLY_SRC} PROPERTY LANGUAGE C)
endif()

if(PLATFORM_ARM32)
    file(GLOB ASSEMBLY_SRC ${NNACL_DIR}/assembly/arm32/*.S)
    set_property(SOURCE ${ASSEMBLY_SRC} PROPERTY LANGUAGE C)
endif()

if("${X86_64_SIMD}" STREQUAL "sse")
    file(GLOB ASSEMBLY_SRC ${NNACL_DIR}/intrinsics/sse/*.c)
    set_property(SOURCE ${ASSEMBLY_SRC} PROPERTY LANGUAGE C)
endif()

if("${X86_64_SIMD}" STREQUAL "avx")
    file(GLOB ASSEMBLY_SRC ${NNACL_DIR}/intrinsics/sse/*.c
            ${NNACL_DIR}/intrinsics/avx/*.c
            ${NNACL_DIR}/assembly/avx/*.S)
    set_property(SOURCE ${ASSEMBLY_SRC} PROPERTY LANGUAGE C)
endif()

if(APPLE)
    set_source_files_properties(${ASSEMBLY_SRC} PROPERTIES COMPILE_FLAGS "-x assembler-with-cpp")
endif()

########################### build nnacl static library ########################
string(REPLACE "-fvisibility=hidden" "-fvisibility=default" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
add_library(nnacl STATIC ${KERNEL_SRC} ${TRAIN_SRC} ${ASSEMBLY_SRC})
add_library(nnacl_mid OBJECT ${KERNEL_SRC} ${TRAIN_SRC} ${ASSEMBLY_SRC})
add_dependencies(nnacl fbs_src)
add_dependencies(nnacl_mid fbs_src)

########################### arm64 build optimize library ########################
if(PLATFORM_ARM64)
    add_subdirectory(${NNACL_DIR}/optimize)
endif()
