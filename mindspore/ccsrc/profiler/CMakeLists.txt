if(ENABLE_GPU)
    file(GLOB_RECURSE PROFILER_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
      "device/gpu/*.cc" "device/cpu/*.cc" "device/profiling.cc" "device/data_saver.cc")
    set_property(SOURCE ${PROFILER_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
      SUBMODULE_ID=mindspore::SubModuleId::SM_PROFILER)
    add_library(_mindspore_profiler_obj OBJECT ${PROFILER_SRC_LIST})
endif()

if(ENABLE_D)
    file(GLOB_RECURSE PROFILER_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
      "device/common/*.cc" "device/ascend/*.cc" "device/cpu/*.cc" "device/profiling.cc" "device/data_saver.cc")
    set_property(SOURCE ${PROFILER_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
      SUBMODULE_ID=mindspore::SubModuleId::SM_PROFILER)
    add_library(_mindspore_profiler_obj OBJECT ${PROFILER_SRC_LIST})
    add_dependencies(_mindspore_profiler_obj mindspore::protobuf)
endif()

if(ENABLE_CPU AND NOT (ENABLE_D OR ENABLE_GPU))
    file(GLOB_RECURSE PROFILER_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
      "device/cpu/*.cc" "device/profiling.cc" "device/data_saver.cc")
    set_property(SOURCE ${PROFILER_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
      SUBMODULE_ID=mindspore::SubModuleId::SM_PROFILER)
    add_library(_mindspore_profiler_obj OBJECT ${PROFILER_SRC_LIST})
endif()