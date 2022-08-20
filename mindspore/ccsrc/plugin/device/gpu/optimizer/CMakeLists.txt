if("${ENABLE_HIDDEN}" STREQUAL "OFF" AND NOT MSVC)
    string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

file(GLOB_RECURSE _PREACTIVATE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
file(GLOB_RECURSE _TRT_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "trt_pass/*.cc")
list(REMOVE_ITEM _PREACTIVATE_SRC_LIST ${_TRT_SRC_LIST})

if(ENABLE_GPU_INFER)
    file(GLOB_RECURSE GPU_SRC_TRT_PASS_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "trt_pass/*.cc")
    list(APPEND _PREACTIVATE_SRC_LIST ${GPU_SRC_TRT_PASS_LIST})
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -Wno-user-defined-warnings -Wno-inconsistent-missing-override \
        -Wno-overloaded-virtual -Wno-unused-const-variable -Wno-pessimizing-move")
endif()

set_property(SOURCE ${_PREACTIVATE_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
  SUBMODULE_ID=mindspore::SubModuleId::SM_PRE_ACT)
if(ENABLE_GPU)
    add_library(_mindspore_plugin_device_gpu_optimizer_obj OBJECT ${_PREACTIVATE_SRC_LIST})
endif()
