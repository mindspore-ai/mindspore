## make protobuf files
# refresh the onnx sha256sum
__download_pkg(onnx https://gitee.com/mirrors/ONNX/repository/archive/v1.6.0
56a2606f684821213ad2a61713521eb9027bd8ac05520be3fb26e997787cedd7)
file(COPY ${onnx_SOURCE_DIR}/onnx/onnx.proto DESTINATION ${TOP_DIR}/third_party/proto/onnx)
file(GLOB ONNX_PROTO "" ${CMAKE_SOURCE_DIR}/third_party/proto/onnx/onnx.proto)
message("onnx proto path is :" ${ONNX_PROTO})
ms_protobuf_generate(ONNX_PROTO_SRCS ONNX_PROTO_HDRS ${ONNX_PROTO})
list(APPEND MINDSPORE_PROTO_LIST ${ONNX_PROTO_SRCS})

include_directories("${CMAKE_BINARY_DIR}/cluster/rpc/core/")
file(GLOB_RECURSE COMM_PROTO_IN RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "cluster/rpc/core/protos/*.proto")
ms_protobuf_generate(COMM_PROTO_SRCS COMM_PROTO_HDRS ${COMM_PROTO_IN})
list(APPEND MINDSPORE_PROTO_LIST ${COMM_PROTO_SRCS})

include_directories("${CMAKE_BINARY_DIR}/cluster/topology")
file(GLOB_RECURSE DISTRIBUTED_CLUSTER_TOPOLOGY RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "cluster/topology/protocol/*.proto")
ms_protobuf_generate(DISTRIBUTED_CLUSTER_TOPOLOGY_SRCS DISTRIBUTED_CLUSTER_TOPOLOGY_HDRS
        ${DISTRIBUTED_CLUSTER_TOPOLOGY})
list(APPEND MINDSPORE_PROTO_LIST ${DISTRIBUTED_CLUSTER_TOPOLOGY_SRCS})

if(ENABLE_DEBUGGER)
    # debugger: compile proto files
    include_directories("${CMAKE_BINARY_DIR}/tools/data_dump/debugger")
    file(
        GLOB_RECURSE DEBUGGER_PROTO_LIST
        RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "tools/data_dump/debugger/debug_graph.proto"
    )
    ms_protobuf_generate(DEBUGGER_PROTO_SRCS DEBUGGER_PROTO_HDRS ${DEBUGGER_PROTO_LIST})
    list(APPEND MINDSPORE_PROTO_LIST ${DEBUGGER_PROTO_SRCS})
endif()

if(ENABLE_DUMP_PROTO)
    include_directories(${CMAKE_BINARY_DIR})

    file(GLOB_RECURSE PROTO_PY RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            "utils/anf_ir.proto"
            "utils/summary.proto"
            "utils/lineage.proto"
            "utils/checkpoint.proto"
            "utils/print.proto"
            "utils/node_strategy.proto"
            "utils/profiling_parallel.proto"
            )
    ms_protobuf_generate_py(PY_SRCS PY_HDRS PY_PYS ${PROTO_PY})

    list(APPEND MINDSPORE_PROTO_LIST ${PROTO_SRCS})
    list(APPEND MINDSPORE_PROTO_LIST ${PY_SRCS})
endif()

# core/mindir.proto
file(GLOB_RECURSE CORE_PROTO_IN ${CMAKE_SOURCE_DIR}/mindspore/core/proto/*.proto)
ms_protobuf_generate(CORE_PROTO_SRC CORE_PROTO_HDR ${CORE_PROTO_IN})
list(APPEND MINDSPORE_PROTO_LIST ${CORE_PROTO_SRC})

if(MINDSPORE_PROTO_LIST)
    add_library(proto_input STATIC ${MINDSPORE_PROTO_LIST})
    target_link_libraries(proto_input PRIVATE mindspore::protobuf)
    if(NOT MSVC)
        set_target_properties(proto_input PROPERTIES COMPILE_FLAGS "-Wno-unused-variable -Wno-array-bounds")
    endif()
endif()
