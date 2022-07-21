if(ENABLE_D OR ENABLE_ACL)
    file(GLOB_RECURSE _TRANSFORM_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
    if(BUILD_LITE)
        list(REMOVE_ITEM _TRANSFORM_SRC_LIST "callbacks_ge.cc")
    endif()
    set_property(SOURCE ${_TRANSFORM_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
      SUBMODULE_ID=mindspore::SubModuleId::SM_GE_ADPT)
    add_library(_mindspore_transform_graph_ir_obj OBJECT ${_TRANSFORM_SRC_LIST})
endif()
