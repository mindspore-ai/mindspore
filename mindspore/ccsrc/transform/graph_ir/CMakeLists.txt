if(ENABLE_GE OR ENABLE_D OR ENABLE_ACL)
    file(GLOB_RECURSE _TRANSFORM_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
    list(REMOVE_ITEM _TRANSFORM_SRC_LIST "graph_ir/op_declare/hcom_ops_declare.cc")
    set_property(SOURCE ${_TRANSFORM_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
      SUBMODULE_ID=mindspore::SubModuleId::SM_GE_ADPT)
    add_library(_mindspore_transform_graph_ir_obj OBJECT ${_TRANSFORM_SRC_LIST})

    if(NOT ENABLE_GE)
        target_compile_definitions(_mindspore_transform_graph_ir_obj PRIVATE NO_GE_CLIENT)
    endif()
endif()
