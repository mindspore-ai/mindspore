file(GLOB PROTO_FILE "" ${TOP_DIR}/third_party/proto/coreml/*.proto)
ms_protobuf_generate(PROTO_SRCS PROTO_HDRS ${PROTO_FILE})
add_library(coreml_proto_mid OBJECT ${PROTO_SRCS})
include_directories(${CMAKE_BINARY_DIR}/proto)

file(GLOB_RECURSE COREML_RUNTIME_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/coreml_graph.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/op/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/pass/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../delegate_utils.cc
        )
add_library(coreml_kernel_mid OBJECT ${COREML_RUNTIME_SRC})
add_dependencies(coreml_kernel_mid fbs_src)
target_link_libraries(coreml_kernel_mid coreml_proto_mid)
