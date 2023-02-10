file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)
set(DATASET_ENGINE_GNN_SRC_FILES
    graph_data_impl.cc
    graph_data_client.cc
    graph_data_server.cc
    graph_loader.cc
    graph_loader_array.cc
    graph_feature_parser.cc
    local_node.cc
    local_edge.cc
    feature.cc
)

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    add_library(engine-gnn OBJECT ${DATASET_ENGINE_GNN_SRC_FILES})
else()
    set(DATASET_ENGINE_GNN_SRC_FILES
        ${DATASET_ENGINE_GNN_SRC_FILES}
        tensor_proto.cc
        grpc_async_server.cc
        graph_data_service_impl.cc
        graph_shared_memory.cc)

    ms_protobuf_generate(TENSOR_PROTO_SRCS TENSOR_PROTO_HDRS "gnn_tensor.proto")
    ms_grpc_generate(GNN_PROTO_SRCS GNN_PROTO_HDRS "gnn_graph_data.proto")

    add_library(engine-gnn OBJECT ${TENSOR_PROTO_SRCS} ${GNN_PROTO_SRCS} ${DATASET_ENGINE_GNN_SRC_FILES})
    add_dependencies(engine-gnn mindspore::protobuf)
endif()
