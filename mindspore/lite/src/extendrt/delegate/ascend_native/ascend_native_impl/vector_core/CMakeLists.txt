# npu
file(GLOB SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/copy_cast.cc
)
set(ASCEND_CORE_TYPE "VectorCore")
set(ASCEND_TARGET ascend_native_vector)
add_library(${ASCEND_TARGET} SHARED ${SRC})
include(../common.cmake)

