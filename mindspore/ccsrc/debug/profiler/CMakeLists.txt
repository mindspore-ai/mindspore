if(NOT ENABLE_SECURITY)
    list(APPEND PROFILER_SRC_LIST ${CMAKE_CURRENT_SOURCE_DIR}/profiling.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/data_saver.cc)

    set_property(SOURCE ${PROFILER_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
            SUBMODULE_ID=mindspore::SubModuleId::SM_PROFILER)
    add_library(_mindspore_debug_profiler_obj OBJECT ${PROFILER_SRC_LIST})
endif()