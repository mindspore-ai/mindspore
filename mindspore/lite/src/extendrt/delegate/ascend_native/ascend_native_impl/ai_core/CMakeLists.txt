# npu


set(SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cc
)

set(ASCEND_CORE_TYPE "AiCore")
set(ASCEND_TARGET ascend_native_ai)
add_library(${ASCEND_TARGET} SHARED ${SRC})
include(../common.cmake)