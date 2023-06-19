if(ENABLE_D OR ENABLE_ACL)
    file(GLOB _ACL_IR_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
    file(GLOB_RECURSE _ACL_DECLARE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "./acl_declare/*.cc")
    list(REMOVE_ITEM _ACL_IR_SRC_LIST "acl_adapter_info.cc")
    list(APPEND _ACL_DECLARE_SRC_LIST "acl_adapter_info.cc")
    set_property(SOURCE ${_ACL_IR_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
      SUBMODULE_ID=mindspore::SubModuleId::SM_GE_ADPT)
    set_property(SOURCE ${_ACL_DECLARE_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
      SUBMODULE_ID=mindspore::SubModuleId::SM_GE_ADPT)
    add_library(_mindspore_transform_acl_ir_obj OBJECT ${_ACL_IR_SRC_LIST})
    add_library(_mindspore_transform_acl_declare_obj OBJECT ${_ACL_DECLARE_SRC_LIST})
endif()