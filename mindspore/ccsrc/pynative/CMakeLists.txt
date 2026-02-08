file(GLOB _PYNATIVE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "forward/*.cc"
        "forward/op_graph/*.cc"
        "forward/pyboost/*.cc"
        "forward/pyboost/customize/*.cc"
        "forward/pyboost/auto_generate/*.cc"
        "backward/*.cc"
        "backward/hook/*.cc"
        "backward/jit_grad/*.cc"
        "backward/op_grad/*.cc"
        "backward/op_grad/auto_generate/*.cc"
        "utils/*.cc"
        "parallel/*.cc"
        "*.cc"
)
set_property(SOURCE ${_PYNATIVE_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_PYNATIVE)

if("${ENABLE_HIDDEN}" STREQUAL "OFF" AND NOT MSVC)
    string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    add_library(mindspore_pynative SHARED ${_PYNATIVE_SRC_LIST})
    target_compile_definitions(mindspore_pynative PRIVATE PYNATIVE_DLL)
else()
    add_library(_mindspore_pynative_obj OBJECT ${_PYNATIVE_SRC_LIST})
    list(APPEND PYNATIVE_OBJ_SRC $<TARGET_OBJECTS:_mindspore_pynative_obj>)
    add_library(mindspore_pynative SHARED ${PYNATIVE_OBJ_SRC})
endif()
target_link_libraries(mindspore_pynative PRIVATE mindspore_core mindspore_ops mindspore_frontend mindspore_pyboost
                      mindspore_common mindspore::pybind11_module mindspore_memory_pool
                      mindspore_profiler mindspore_runtime_pipeline mindspore_hardware_abstract
                      mindspore_backend_common mindspore_pynative_utils mindspore_runtime_utils)

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set_target_properties(mindspore_pynative PROPERTIES MACOSX_RPATH ON)
    set_target_properties(mindspore_pynative PROPERTIES INSTALL_RPATH @loader_path)
else()
    set_target_properties(mindspore_pynative PROPERTIES INSTALL_RPATH $ORIGIN)
endif()

add_dependencies(mindspore_pynative generated_code)
