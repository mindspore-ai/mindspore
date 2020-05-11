if (ENABLE_GE OR ENABLE_D)
    file(GLOB_RECURSE _TRANSFORM_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
    set_property(SOURCE ${_TRANSFORM_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_GE_ADPT)
    add_library(_mindspore_transform_obj OBJECT ${_TRANSFORM_SRC_LIST})

    if (NOT ENABLE_GE)
        target_compile_definitions(_mindspore_transform_obj PRIVATE NO_GE_CLIENT)
    endif()
endif ()
