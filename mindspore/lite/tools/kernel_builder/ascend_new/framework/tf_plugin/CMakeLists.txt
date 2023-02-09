# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
aux_source_directory(. SRCS)
message(STATUS "SRCS = ${SRCS}")

if("x${SRCS}" STREQUAL "x")
    add_custom_target(${TF_PLUGIN_TARGET}
            COMMAND mkdir -p ${TF_PLUGIN_TARGET_OUT_DIR}
            COMMAND echo "no source to make lib${TF_PLUGIN_TARGET}.so")
    return(0)
endif()

set(LIBRARY_OUTPUT_PATH ${TF_PLUGIN_TARGET_OUT_DIR})

add_library(${TF_PLUGIN_TARGET} SHARED ${SRCS})

target_compile_definitions(${TF_PLUGIN_TARGET} PRIVATE
    google=ascend_private
)

target_link_libraries(${TF_PLUGIN_TARGET} ${ASCEND_INC}/../lib64/libgraph.so)
