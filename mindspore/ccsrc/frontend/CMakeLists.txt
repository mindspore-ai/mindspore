## make sub objects
set(SUB_COMP
    expander
    jit/ps
    jit/pi
    jit/trace
    operator
    optimizer
    parallel
)

foreach(_comp ${SUB_COMP})
    add_subdirectory(${_comp})
    string(REPLACE "/" "_" sub ${_comp})
    if(TARGET _mindspore_frontend_${sub}_obj)
        list(APPEND SUB_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_frontend_${sub}_obj>)
        add_dependencies(_mindspore_frontend_${sub}_obj proto_input mindspore_core mindspore_ops mindspore_ms_backend)
    endif()
    if(CMAKE_SYSTEM_NAME MATCHES "Windows")
        target_compile_definitions(_mindspore_frontend_${sub}_obj PRIVATE FRONTEND_DLL)
    endif()
endforeach()

set_property(SOURCE ${SUB_OBJECTS_SRC} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_ME)
add_library(mindspore_frontend SHARED ${SUB_OBJECTS_SRC})
target_link_libraries(mindspore_frontend PRIVATE mindspore_core mindspore::pybind11_module mindspore::securec
                      mindspore_common mindspore_ops mindspore_backend_manager proto_input mindspore_profiler
                      mindspore_runtime_pipeline mindspore_backend_common mindspore_hardware_abstract
                      mindspore_runtime_utils mindspore_cluster)
if(NOT WIN32)
    target_link_libraries(mindspore_frontend PRIVATE dl)
endif()
