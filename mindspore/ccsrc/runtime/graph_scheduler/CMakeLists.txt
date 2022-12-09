include_directories(${CMAKE_SOURCE_DIR}/mindspore/core/mindrt/include)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core/mindrt/src)

file(GLOB_RECURSE GRAPH_SCHEDULER_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

if(NOT ENABLE_CPU OR WIN32 OR APPLE)
  list(REMOVE_ITEM GRAPH_SCHEDULER_SRC_LIST "rpc_node_scheduler.cc")
  list(REMOVE_ITEM GRAPH_SCHEDULER_SRC_LIST "actor/rpc/recv_actor.cc")
  list(REMOVE_ITEM GRAPH_SCHEDULER_SRC_LIST "actor/rpc/rpc_actor.cc")
  list(REMOVE_ITEM GRAPH_SCHEDULER_SRC_LIST "actor/rpc/send_actor.cc")
  list(REMOVE_ITEM GRAPH_SCHEDULER_SRC_LIST "actor/rpc/mux_send_actor.cc")
  list(REMOVE_ITEM GRAPH_SCHEDULER_SRC_LIST "actor/rpc/mux_recv_actor.cc")
  list(REMOVE_ITEM GRAPH_SCHEDULER_SRC_LIST "actor/embedding_cache/embedding_cache_prefetch_actor.cc")
  list(REMOVE_ITEM GRAPH_SCHEDULER_SRC_LIST "actor/embedding_cache/device_embedding_operation.cc")
  list(REMOVE_ITEM GRAPH_SCHEDULER_SRC_LIST "actor/embedding_cache/device_dense_embedding_operation.cc")
  list(REMOVE_ITEM GRAPH_SCHEDULER_SRC_LIST "actor/embedding_cache/device_sparse_embedding_operation.cc")
  list(REMOVE_ITEM GRAPH_SCHEDULER_SRC_LIST "actor/embedding_cache/device_sparse_embedding_operation.cc")
  list(REMOVE_ITEM GRAPH_SCHEDULER_SRC_LIST "embedding_cache_scheduler.cc")
endif()

set_property(SOURCE ${GRAPH_SCHEDULER_SRC_LIST}
  PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_RUNTIME_FRAMEWORK)

add_library(_mindspore_runtime_graph_scheduler_obj OBJECT ${GRAPH_SCHEDULER_SRC_LIST})
