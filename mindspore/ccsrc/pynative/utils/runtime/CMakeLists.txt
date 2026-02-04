file(GLOB_RECURSE PYNATIVE_UTILS_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "*.cc"
        )

set_property(SOURCE ${PYNATIVE_UTILS_SRC_LIST}
        PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_PYNATIVE)
add_library(mindspore_pynative_utils SHARED ${PYNATIVE_UTILS_SRC_LIST})
target_link_libraries(mindspore_pynative_utils PRIVATE mindspore_core
        mindspore_common mindspore_hardware_abstract mindspore_profiler mindspore_runtime_pipeline
        mindspore_backend_common mindspore_memory_pool mindspore_runtime_utils mindspore::pybind11_module)
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    target_compile_definitions(mindspore_pynative_utils PRIVATE PYNATIVE_UTILS_DLL)
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set_target_properties(mindspore_pynative_utils PROPERTIES MACOSX_RPATH ON)
    set_target_properties(mindspore_pynative_utils PROPERTIES INSTALL_RPATH @loader_path)
else()
    set_target_properties(mindspore_pynative_utils PROPERTIES INSTALL_RPATH $ORIGIN)
endif()