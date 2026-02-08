file(GLOB_RECURSE _CLUSTER_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

if(NOT ENABLE_CPU OR WIN32 OR APPLE)
    list(REMOVE_ITEM _CLUSTER_SRC_FILES "topology/cluster_context.cc")
    list(REMOVE_ITEM _CLUSTER_SRC_FILES "topology/actor_route_table_proxy.cc")
    list(REMOVE_ITEM _CLUSTER_SRC_FILES "topology/compute_graph_node.cc")
    list(REMOVE_ITEM _CLUSTER_SRC_FILES "topology/meta_server_node.cc")
    list(REMOVE_ITEM _CLUSTER_SRC_FILES "topology/tcp_node.cc")
else()
    list(REMOVE_ITEM _CLUSTER_SRC_FILES "topology/dummy_cluster_context.cc")
endif()


if(NOT ENABLE_CPU OR WIN32 OR APPLE)
    set(EXCLUDE_DIRS "rpc/")
    foreach(EXCLUDE_DIR ${EXCLUDE_DIRS})
        foreach(EXCLUDE_FILE ${_CLUSTER_SRC_FILES})
            string(FIND ${EXCLUDE_FILE} ${EXCLUDE_DIR} FOUND)
            if(${FOUND} EQUAL 0)
                list(REMOVE_ITEM _CLUSTER_SRC_FILES ${EXCLUDE_FILE})
            endif()
        endforeach()
    endforeach()
    list(APPEND _CLUSTER_SRC_FILES "rpc/core/ps_context.cc")
endif()

set_property(SOURCE ${_CLUSTER_SRC_FILES} PROPERTY COMPILE_DEFINITIONS
            SUBMODULE_ID=mindspore::SubModuleId::SM_DISTRIBUTED)
add_library(mindspore_cluster SHARED ${_CLUSTER_SRC_FILES})
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    target_compile_definitions(mindspore_cluster PRIVATE CLUSTER_DLL)
endif()

target_link_libraries(mindspore_cluster PRIVATE proto_input mindspore_core
                      mindspore_hardware_abstract mindspore_runtime_pipeline
                      mindspore_common mindspore::securec)

if(ENABLE_CPU AND NOT WIN32)
    target_link_libraries(mindspore_cluster PRIVATE mindspore::ssl mindspore::crypto
                          mindspore::event mindspore::event_pthreads
                          mindspore::event_openssl mindspore::event_core)
endif()
