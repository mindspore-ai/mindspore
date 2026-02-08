file(GLOB HOST_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "*.cc"
        "view/*.cc"
        "view/kernel_mod_impl/*.cc"
)

set_property(SOURCE ${HOST_SRC_LIST}
        PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_KERNEL)

add_library(mindspore_ops_host SHARED ${HOST_SRC_LIST})

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    target_compile_definitions(mindspore_ops_host PRIVATE KERNEL_HOST_DLL)
endif()

target_link_libraries(mindspore_ops_host PRIVATE mindspore_hardware_abstract
        mindspore_core mindspore_ops mindspore_common mindspore_backend_common)

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set_target_properties(mindspore_ops_host PROPERTIES MACOSX_RPATH ON)
    set_target_properties(mindspore_ops_host PROPERTIES INSTALL_RPATH @loader_path)
else()
    set_target_properties(mindspore_ops_host PROPERTIES INSTALL_RPATH $ORIGIN)
endif()
