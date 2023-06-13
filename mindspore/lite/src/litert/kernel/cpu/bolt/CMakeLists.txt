SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-variable -Wno-unused-but-set-variable")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable -Wno-unused-but-set-variable")
set(BOLT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/bolt)

cmake_policy(SET CMP0077 NEW)
set(USE_GENERAL ON)
set(USE_FP32 ON)
set(BOLT_FLAGS "${BOLT_FLAGS} -D_USE_GENERAL -D_USE_FP32")

if(PLATFORM_X86_64 OR MSLITE_ENABLE_SEE OR MSLITE_ENABLE_AVX OR MSLITE_ENABLE_AVX512)
    set(USE_X86 ON)
    set(BOLT_FLAGS "${BOLT_FLAGS} -D_USE_X86")
endif()

set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${BOLT_FLAGS}")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${BOLT_FLAGS}")

add_subdirectory(${BOLT_ROOT}/compute)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/bolt/common/uni/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/bolt/common/memory/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/bolt/compute/tensor/include)

file(GLOB KERNEL_BOLT_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
        )

add_library(bolt_kernel_mid OBJECT ${KERNEL_BOLT_SRC})
add_dependencies(bolt_kernel_mid fbs_src)

target_link_libraries(bolt_kernel_mid blas_enhance)
target_link_libraries(bolt_kernel_mid tensor)
target_link_libraries(bolt_kernel_mid image)
