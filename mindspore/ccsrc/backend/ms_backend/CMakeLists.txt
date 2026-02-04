file(GLOB_RECURSE MS_BACKEND_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

set_property(SOURCE ${MS_BACKEND_SRC}
        PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_RUNTIME_FRAMEWORK)

if("${ENABLE_HIDDEN}" STREQUAL "OFF" AND NOT MSVC)
  string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

include_directories(${CMAKE_SOURCE_DIR}/mindspore/core/mindrt)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core/mindrt/include)

add_library(mindspore_ms_backend SHARED ${MS_BACKEND_SRC})

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
        target_compile_definitions(mindspore_ms_backend PRIVATE BACKEND_DLL)
endif()

target_link_libraries(mindspore_ms_backend PRIVATE mindspore::pybind11_module)
target_link_libraries(mindspore_ms_backend PRIVATE mindspore_core mindspore_ops
        mindspore_common mindspore_backend_manager proto_input mindspore_hardware_abstract
        mindspore_profiler mindspore_runtime_pipeline mindspore_backend_common mindspore_memory_pool
        mindspore_runtime_utils)

target_link_libraries(mindspore_ms_backend PRIVATE mindspore::securec)
if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set_target_properties(mindspore_ms_backend PROPERTIES MACOSX_RPATH ON)
    set_target_properties(mindspore_ms_backend PROPERTIES INSTALL_RPATH @loader_path)
else()
    set_target_properties(mindspore_ms_backend PROPERTIES INSTALL_RPATH $ORIGIN)
endif()