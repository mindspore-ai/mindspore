# path variables for metadef submodule, it has to be included after mindspore/core
# and minspore/ccsrc to prevent conflict of op headers

set(METADEF_PATH "${CMAKE_SOURCE_DIR}/metadef")

if(ENABLE_TESTCASES OR ENABLE_D OR ENABLE_ACL)
    message("Note: compile cpp with include file: ${ASCEND_PATH}/include/")
    include_directories(${ASCEND_PATH}/cann/include/)
    include_directories(${ASCEND_PATH}/cann/include/hccl)
    include_directories(${ASCEND_PATH}/cann/include/aoe)
    include_directories(${ASCEND_PATH}/cann/include/exe_graph)
    include_directories(${ASCEND_PATH}/cann/opp/built-in/)
    include_directories(${ASCEND_PATH}/cann/opp/built-in/op_impl/aicpu/aicpu_kernel/inc/)
    include_directories(${ASCEND_PATH}/cann/pkg_inc/)
    include_directories(${ASCEND_PATH}/cann/pkg_inc/runtime/)
    include_directories(${ASCEND_PATH}/cann/pkg_inc/profiling)
    include_directories(${METADEF_PATH}/inc/)
    include_directories(${METADEF_PATH}/inc/external/)
endif()