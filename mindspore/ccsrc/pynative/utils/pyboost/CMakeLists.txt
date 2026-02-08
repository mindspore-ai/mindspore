file(GLOB_RECURSE _PYBOOST_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "auto_generate/*.cc"
        "customize/*.cc"
        "functions/composite/*.cc"
        "functions/*.cc"
        "functions/auto_generate/*.cc"
        "grad_functions/*.cc"
        "grad_functions/auto_generate/*.cc"
        "comm_handle.cc"
        "comm_utils.cc"
        "pyboost_utils.cc"
        "kernel_mod_cache.cc"
)

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    file(GLOB_RECURSE _CUSTOM_SRC "custom/*.cc")
    list(APPEND _PYBOOST_SRC_LIST ${_CUSTOM_SRC})
endif()

set_property(SOURCE ${_PYBOOST_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_PYNATIVE)
add_library(mindspore_pyboost SHARED ${_PYBOOST_SRC_LIST})
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    target_compile_definitions(mindspore_pyboost PRIVATE PYBOOST_DLL)
endif()
target_link_libraries(mindspore_pyboost PRIVATE mindspore_core mindspore_ops mindspore_common
                      mindspore_hardware_abstract mindspore::pybind11_module mindspore_profiler
                      mindspore_runtime_pipeline mindspore_backend_common mindspore_runtime_utils
                      mindspore_pynative_utils mindspore_memory_pool)
add_dependencies(mindspore_pyboost generated_code)
