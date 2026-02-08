file(GLOB_RECURSE RUNTIME_COMMON_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

if("${ENABLE_HIDDEN}" STREQUAL "OFF" AND NOT MSVC)
  string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

set_property(SOURCE ${RUNTIME_COMMON_LIST} PROPERTY SUBMODULE_ID=mindspore::SubModuleId::SM_RUNTIME_FRAMEWORK)
add_library(mindspore_runtime_utils SHARED ${RUNTIME_COMMON_LIST})
target_link_libraries(mindspore_runtime_utils PRIVATE mindspore_core)
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
        target_compile_definitions(mindspore_runtime_utils PRIVATE RUNTIME_UTILS_DLL)
endif()
set_target_properties(mindspore_runtime_utils PROPERTIES INSTALL_RPATH $ORIGIN)
