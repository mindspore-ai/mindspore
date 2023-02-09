if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    file(GLOB_RECURSE _COMMON_ALL_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "duplex_pipe_win.cc"
        "thread_pool.cc"
        "fallback.cc"
    )
else()
    file(GLOB_RECURSE _COMMON_ALL_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "duplex_pipe.cc"
        "thread_pool.cc"
        "fallback.cc"
    )
endif()

set_property(SOURCE ${_COMMON_ALL_SRC_FILES} PROPERTY COMPILE_DEFINITIONS
  SUBMODULE_ID=mindspore::SubModuleId::SM_COMMON)
add_library(_mindspore_common_obj OBJECT ${_COMMON_ALL_SRC_FILES})
