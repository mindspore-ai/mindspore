if(${PYTHON_VERSION} VERSION_LESS "3.12")
    file(GLOB_RECURSE _PIJIT_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            "graph_capture/*.cc"
            "graph_compiler/*.cc"
            "graph_guard/*.cc"
            "graph_build/*.cc"
            "utils/*.cc"
            "*.c"
            "*.cc"
            "python_adapter/*.c"
            "python_adapter/*.cc"
    )
else()
    # pijit cannot support python>=3.12 for now, but will be adapted very soon.
    file(GLOB_RECURSE _PIJIT_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            "init.cc"
    )
endif()

set_property(SOURCE ${_PIJIT_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_PI)
add_library(_mindspore_frontend_jit_pi_obj OBJECT ${_PIJIT_SRC_LIST})
if(NOT MSVC)
    target_compile_options(_mindspore_frontend_jit_pi_obj PRIVATE -fPIC)
endif()
