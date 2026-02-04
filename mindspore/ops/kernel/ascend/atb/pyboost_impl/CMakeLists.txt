if(EXISTS ${ASCEND_NNAL_ATB_PATH})
    include_directories(${ASCEND_NNAL_ATB_PATH}/include)
    file(GLOB_RECURSE PYBOOST_ATB_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        atb_runner.cc
        ../kernel_mod_impl/atb_adapter.cc
    )

    add_library(mindspore_pyboost_atb_kernels SHARED ${PYBOOST_ATB_SRCS})
    find_library(ATB_KENREL atb ${ASCEND_NNAL_ATB_OPP_PATH})
    target_link_libraries(mindspore_pyboost_atb_kernels PRIVATE ${ATB_KENREL} mindspore_backend_common
        mindspore_pyboost mindspore_ascend)
    set_target_properties(mindspore_pyboost_atb_kernels PROPERTIES INSTALL_RPATH
        $ORIGIN:$ORIGIN/../:$ORIGIN/../../:$ORIGIN/../../../)
endif()