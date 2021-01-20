if(ENABLE_GPU)
    file(GLOB_RECURSE PROFILER_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "device/gpu/*.cc")
    set_property(SOURCE ${PROFILER_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
      SUBMODULE_ID=mindspore::SubModuleId::SM_PROFILER)
    add_library(_mindspore_profiler_obj OBJECT ${PROFILER_SRC_LIST})
endif()

if(ENABLE_D)
    file(GLOB_RECURSE PROFILER_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "device/ascend/*.cc" "device/common/*.cc")
    set_property(SOURCE ${PROFILER_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
      SUBMODULE_ID=mindspore::SubModuleId::SM_PROFILER)
    add_library(_mindspore_profiler_obj OBJECT ${PROFILER_SRC_LIST})
    add_dependencies(_mindspore_profiler_obj mindspore::protobuf)
endif()
