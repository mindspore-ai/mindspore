if(ENABLE_CPU AND ${CMAKE_SYSTEM_NAME} MATCHES "Linux"
    AND ${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "x86_64")
    message("compiled quantum simulator")
    file(GLOB_RECURSE _MINDQUANTUM_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
    set_property(SOURCE ${_MINDQUANTUM_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
        SUBMODULE_ID=mindspore::SubModuleId::SM_MINDQUANTUM)
    add_library(_mindspore_mindquantum_obj OBJECT ${_MINDQUANTUM_SRC_LIST})
    target_compile_options(_mindspore_mindquantum_obj PRIVATE -fopenmp -mavx -ffast-math)
    target_compile_definitions(_mindspore_mindquantum_obj PRIVATE INTRIN)
endif()