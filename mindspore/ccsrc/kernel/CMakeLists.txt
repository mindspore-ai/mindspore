file(GLOB_RECURSE KERNEL_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "kernel_factory.cc"
    "kernel_build_info.cc"
    "kernel.cc"
    "common_utils.cc"
    "kash/*.cc"
    "oplib/*.cc"
    "environ_manager.cc"
)

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-delete-non-abstract-non-virtual-dtor -Wno-overloaded-virtual")
endif()

if(ENABLE_AKG AND ${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    file(GLOB_RECURSE AKG_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "akg/akg_kernel_build.cc"
        "akg/akg_kernel_build_manager.cc"
        "akg/akg_kernel_json_generator.cc"
        "akg/akg_kernel_json_decoder.cc"
    )
    list(APPEND KERNEL_SRC_LIST "${AKG_SRC_LIST}")
else()
    file(GLOB_RECURSE AKG_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "akg/akg_kernel_json_generator.cc"
        "akg/akg_kernel_json_decoder.cc"
    )
    list(APPEND KERNEL_SRC_LIST "${AKG_SRC_LIST}")
endif()

set_property(SOURCE ${KERNEL_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_KERNEL)
add_library(_mindspore_kernel_obj OBJECT ${KERNEL_SRC_LIST})
