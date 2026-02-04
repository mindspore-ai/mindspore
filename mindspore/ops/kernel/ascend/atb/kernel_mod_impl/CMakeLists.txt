file(GLOB_RECURSE ATB_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "*.cc"
)
list(REMOVE_ITEM ATB_SRC_LIST
    "atb_kernel_build.cc"
)

if(EXISTS ${ASCEND_NNAL_ATB_PATH})
    include_directories(${ASCEND_NNAL_ATB_PATH}/include)
    add_library(mindspore_atb_kernels SHARED ${ATB_SRC_LIST})
    find_library(ATB_KENREL atb ${ASCEND_NNAL_ATB_OPP_PATH})
    target_link_libraries(mindspore_atb_kernels PRIVATE ${ATB_KENREL})
    target_link_libraries(mindspore_atb_kernels PRIVATE mindspore_ascend)
    set_target_properties(mindspore_atb_kernels PROPERTIES INSTALL_RPATH
        $ORIGIN:$ORIGIN/../:$ORIGIN/../../:$ORIGIN/../../../)
endif()