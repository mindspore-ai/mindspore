if(ENABLE_D OR ENABLE_ACL OR ENABLE_TESTCASES)
    file(GLOB _SYMBOL_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
    set_property(SOURCE ${_SYMBOL_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
      SUBMODULE_ID=mindspore::SubModuleId::SM_GE_ADPT)
    add_library(_mindspore_transform_symbol_obj OBJECT ${_SYMBOL_SRC_LIST})
endif()