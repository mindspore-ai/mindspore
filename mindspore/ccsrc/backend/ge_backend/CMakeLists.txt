if(ENABLE_D OR ENABLE_ACL)
    include(${CMAKE_SOURCE_DIR}/cmake/metadef_variables.cmake)

    file(GLOB _GE_BACKEND_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
    list(APPEND _GE_BACKEND_SRC_LIST $<TARGET_OBJECTS:_mindspore_ge_runtime_obj>)
    list(APPEND _GE_BACKEND_SRC_LIST $<TARGET_OBJECTS:_mindspore_ge_dump_obj>)
    list(APPEND _GE_BACKEND_SRC_LIST $<TARGET_OBJECTS:_mindspore_ge_utils_obj>)
    list(APPEND _GE_BACKEND_SRC_LIST $<TARGET_OBJECTS:_mindspore_ge_pass_obj>)
    list(APPEND _GE_BACKEND_SRC_LIST $<TARGET_OBJECTS:_mindspore_ge_executor_obj>)


    set_property(SOURCE ${_GE_BACKEND_SRC_LIST}
          PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_GE_ADPT)
    add_library(mindspore_ge_backend SHARED ${_GE_BACKEND_SRC_LIST})

    target_link_libraries(mindspore_ge_backend PRIVATE mindspore_core mindspore_ops mindspore_ascend_res_manager
            mindspore_common mindspore_backend_manager mindspore_backend_common proto_input mindspore_runtime_pipeline
            mindspore_memory_pool mindspore_graph_ir mindspore_runtime_utils)
    target_link_libraries(mindspore_ge_backend PRIVATE d_collective)

    set_target_properties(mindspore_ge_backend PROPERTIES INSTALL_RPATH $ORIGIN:$ORIGIN/plugin:$ORIGIN/plugin/ascend)

    add_subdirectory(pass)
    add_subdirectory(executor)
    add_subdirectory(runtime)
    add_subdirectory(dump)
    add_subdirectory(utils)
    add_subdirectory(graph_ir)
endif()
