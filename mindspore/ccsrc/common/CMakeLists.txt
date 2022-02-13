if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    file(GLOB_RECURSE _COMMON_ALL_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "duplex_pipe_win.cc"
        "thread_pool.cc"
    )
else()
    file(GLOB_RECURSE _COMMON_ALL_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "duplex_pipe.cc"
        "thread_pool.cc"
    )
endif()

if(ENABLE_AKG AND ${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    file(GLOB_RECURSE _GK_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "graph_kernel/*.cc"
    )
    file(GLOB_RECURSE _GK_LITE_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "graph_kernel/lite_adapter/*.cc"
    )
    list(REMOVE_ITEM _GK_SRC_LIST ${_GK_LITE_LIST})
    list(APPEND _COMMON_ALL_SRC_FILES ${_GK_SRC_LIST})
endif()

set_property(SOURCE ${_COMMON_ALL_SRC_FILES} PROPERTY COMPILE_DEFINITIONS
  SUBMODULE_ID=mindspore::SubModuleId::SM_COMMON)
add_library(_mindspore_common_obj OBJECT ${_COMMON_ALL_SRC_FILES})
