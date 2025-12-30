# path variables for metadef submodule, it has to be included after mindspore/core
# and minspore/ccsrc to prevent conflict of op headers

set(METADEF_PATH "${CMAKE_SOURCE_DIR}/metadef")

if(ENABLE_TESTCASES OR ENABLE_D OR ENABLE_ACL)
    message("Note: compile cpp with include file: ${ASCEND_PATH}/include/")
    # CANN 8.5.0 and above include paths
    include_directories(${ASCEND_PATH}/include/)
    include_directories(${ASCEND_PATH}/include/hccl)
    include_directories(${ASCEND_PATH}/include/aoe)
    include_directories(${ASCEND_PATH}/include/exe_graph)
    include_directories(${ASCEND_PATH}/opp/built-in/)
    include_directories(${ASCEND_PATH}/opp/built-in/op_impl/aicpu/aicpu_kernel/inc/)
    include_directories(${ASCEND_PATH}/pkg_inc/)
    include_directories(${ASCEND_PATH}/pkg_inc/runtime/)
    include_directories(${ASCEND_PATH}/pkg_inc/profiling)
    include_directories(${ASCEND_PATH}/pkg_inc/aicpu_common)
    include_directories(${ASCEND_PATH}/pkg_inc/aicpu/cpu_kernels)
    # CANN 8.3.0 and below include paths
    include_directories(${ASCEND_PATH}/include/experiment)
    include_directories(${ASCEND_PATH}/include/experiment/runtime/)
    include_directories(${ASCEND_PATH}/include/experiment/msprof/)
    # metadef include path
    include_directories(${METADEF_PATH}/inc/)
    include_directories(${METADEF_PATH}/inc/external/)
endif()
