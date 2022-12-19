# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_C_COMPILER gcc)
# add source files
aux_source_directory(. SRCS)

if("x${SRCS}" STREQUAL "x")
    add_custom_target(${OP_PROTO_TARGET}
            COMMAND mkdir -p ${OP_PROTO_TARGET_OUT_DIR}
            COMMAND echo "no source to make lib${OP_PROTO_TARGET}.so")
    return(0)
endif()

set(LIBRARY_OUTPUT_PATH ${OP_PROTO_TARGET_OUT_DIR})

message(STATUS "OP_PROTO_TARGET=${OP_PROTO_TARGET}")
add_library(${OP_PROTO_TARGET} SHARED ${SRCS})

target_link_libraries(${OP_PROTO_TARGET} ${ASCEND_INC}/../lib64/libgraph.so)
