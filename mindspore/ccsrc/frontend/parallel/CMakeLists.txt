file(GLOB_RECURSE _PARALLEL_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

if("${ENABLE_HIDDEN}" STREQUAL "OFF" AND NOT MSVC)
    string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

if(ENABLE_DUMP_PROTO)
    list(REMOVE_ITEM _PARALLEL_SRC_FILES "parallel/strategy_checkpoint/parallel_strategy_checkpoint.cc")
    list(REMOVE_ITEM _PARALLEL_SRC_FILES "parallel/strategy_checkpoint/strategy_checkpoint_info.cc")
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-delete-non-abstract-non-virtual-dtor")
endif()

set_property(SOURCE ${_PARALLEL_SRC_FILES} PROPERTY COMPILE_DEFINITIONS
    SUBMODULE_ID=mindspore::SubModuleId::SM_PARALLEL)
add_library(_mindspore_frontend_parallel_obj OBJECT ${_PARALLEL_SRC_FILES})
