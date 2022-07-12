if(NOT ENABLE_SECURITY AND ENABLE_CPU)
    file(GLOB_RECURSE PROFILER_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

    set_property(SOURCE ${PROFILER_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
            SUBMODULE_ID=mindspore::SubModuleId::SM_PROFILER)
    add_library(_mindspore_plugin_device_cpu_hal_profiler_obj OBJECT ${PROFILER_SRC_LIST})
endif()
