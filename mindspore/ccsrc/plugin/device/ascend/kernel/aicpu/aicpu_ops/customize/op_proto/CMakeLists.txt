set(CMAKE_CXX_COMPILER g++)
set(CMAKE_C_COMPILER gcc)
set(OP_PROTO_TARGET "cust_op_proto")
set(LIBRARY_OUTPUT_PATH ${OP_PROTO_PATH})
if(DEFINED ENV{ASCEND_CUSTOM_PATH})
    set(ASCEND_TENSOR_COMPILER $ENV{ASCEND_CUSTOM_PATH}/latest/compiler)
elseif(EXISTS /usr/local/Ascend/latest/)
    set(ASCEND_TENSOR_COMPILER /usr/local/Ascend/latest/compiler)
elseif(EXISTS /usr/local/Ascend/ascend-toolkit/latest)
    set(ASCEND_TENSOR_COMPILER /usr/local/Ascend/ascend-toolkit/latest/compiler)
else()
    set(ASCEND_TENSOR_COMPILER /usr/local/Ascend/compiler)
endif()

# add source files
aux_source_directory(. SRCS)
set(SRCS ${SRCS}
    ../utils/util.cc
    ../utils/transfer_shape_according_to_format.cc
    ../utils/axis_util.cc
    ../src/error_util.cc
    ../src/op_util.cc)

add_library(${OP_PROTO_TARGET} SHARED ${SRCS})

target_link_libraries(${OP_PROTO_TARGET} ${ASCEND_TENSOR_COMPILER}/lib64/libgraph.so)
