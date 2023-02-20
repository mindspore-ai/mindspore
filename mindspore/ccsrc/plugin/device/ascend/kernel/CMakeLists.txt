if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-delete-non-abstract-non-virtual-dtor -Wno-overloaded-virtual")
endif()

file(GLOB_RECURSE D_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "kernel_query.cc"
        "ascend_kernel_mod.cc"
        "acl/*.cc"
        "tbe/*.cc"
        "host/*.cc"
        "aicpu/*.cc"
        "rts/*.cc"
        "hccl/*.cc"
        "bisheng/bisheng_kernel_build.cc"
        "bisheng/bisheng_kernel_factory.cc"
        "bisheng/custom_bisheng_kernel.cc"
        "bisheng/bisheng_kernel_metadata.cc"
    )
    file(GLOB_RECURSE AICPU_OPS_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "aicpu/aicpu_ops/*.cc"
    )
    list(REMOVE_ITEM D_SRC_LIST ${AICPU_OPS_SRC})
    add_compile_definitions(ENABLE_D)

    if(ENABLE_SECURITY)
        list(REMOVE_ITEM D_SRC_LIST "rts/profiling_kernel_mod.cc")
    endif()

    if(ENABLE_AKG AND ${CMAKE_SYSTEM_NAME} MATCHES "Linux" AND ENABLE_D)
        file(GLOB_RECURSE AKG_D_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "akg/*.cc")
endif()

set_property(SOURCE ${D_SRC_LIST} ${AKG_D_SRC_LIST}
    PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_KERNEL)
if(ENABLE_D)
    add_library(_mindspore_plugin_device_ascend_kernel_obj OBJECT ${D_SRC_LIST} ${AKG_D_SRC_LIST})
endif()

add_subdirectory(bisheng)
