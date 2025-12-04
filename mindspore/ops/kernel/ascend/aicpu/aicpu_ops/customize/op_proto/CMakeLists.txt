set(CMAKE_CXX_COMPILER g++)
set(CMAKE_C_COMPILER gcc)
set(OP_PROTO_TARGET "cust_op_proto")
set(LIBRARY_OUTPUT_PATH ${OP_PROTO_PATH})
if(DEFINED ENV{ASCEND_CUSTOM_PATH})
    set(ASCEND_TENSOR_COMPILER $ENV{ASCEND_CUSTOM_PATH}/cann/compiler)
else()
    set(ASCEND_TENSOR_COMPILER /usr/local/Ascend/cann/compiler)
endif()

# add source files
aux_source_directory(. SRCS)
aux_source_directory(utils UTILS_SRCS)

include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/transform/graph_ir/)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/plugin/ascend/res_manager/op_adapter/)

add_library(${OP_PROTO_TARGET} SHARED ${SRCS} ${UTILS_SRCS})

find_library(register libregister.so ${ASCEND_CANN_RUNTIME_PATH})
find_library(graph libgraph.so ${ASCEND_CANN_RUNTIME_PATH})
target_link_libraries(${OP_PROTO_TARGET} ${register})
target_link_libraries(${OP_PROTO_TARGET} ${graph})
