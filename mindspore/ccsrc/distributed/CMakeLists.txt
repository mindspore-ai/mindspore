file(GLOB_RECURSE _DISTRIBUTED_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

if(NOT ENABLE_CPU OR WIN32)
    list(REMOVE_ITEM _DISTRIBUTED_SRC_FILES "cluster/cluster_context.cc")
else()
    list(REMOVE_ITEM _DISTRIBUTED_SRC_FILES "cluster/dummy_cluster_context.cc")
endif()

set_property(SOURCE ${_DISTRIBUTED_SRC_FILES} PROPERTY COMPILE_DEFINITIONS
            SUBMODULE_ID=mindspore::SubModuleId::SM_DISTRIBUTED)
add_library(_mindspore_distributed_obj OBJECT ${_DISTRIBUTED_SRC_FILES})