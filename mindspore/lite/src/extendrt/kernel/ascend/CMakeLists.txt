include_directories(${TOP_DIR}/graphengine/910/inc/external)
include_directories(${TOP_DIR}/mindspore)
include_directories(${TOP_DIR}/mindspore/lite/src)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath,$ORIGIN/")

find_library(ge_graph libgraph.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
find_library(acl libascendcl.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
find_library(acl_cblas libacl_cblas.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})

file(GLOB_RECURSE ASCEND_SRC ${CMAKE_CURRENT_SOURCE_DIR}
        "src/*.cc"
        "api/*.cc"
        "model/*.cc"
        )

add_library(ascend_kernel_plugin SHARED ${ASCEND_SRC})
add_dependencies(ascend_kernel_plugin fbs_inner_src)
add_dependencies(ascend_kernel_plugin mindspore-extendrt)
target_link_libraries(ascend_kernel_plugin mindspore-extendrt)

if("${MSLITE_REGISTRY_DEVICE}" STREQUAL "SD3403" AND PLATFORM_ARM64)
    find_library(acl_retr libacl_retr.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(acl_runtime libruntime.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    target_link_libraries(ascend_kernel_plugin ${ge_graph} ${acl} ${acl_retr} ${acl_cblas} ${acl_runtime})
else()
    find_library(acl_dvpp libacl_dvpp.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(ge_compiler libge_compiler.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(libplatform libplatform.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(libcompress libcompress.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(libopskernel libopskernel.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(libaicore_utils libaicore_utils.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(libaicpu_engine_common libaicpu_engine_common.so ${ASCEND_CANN_RUNTIME_PATH}
        ${ASCEND_TOOLKIT_RUNTIME_PATH})
    target_link_libraries(ascend_kernel_plugin ${ge_graph} ${ge_compiler} ${acl_cblas} ${acl_dvpp} ${acl_runtime}
        ${libplatform} ${libcompress} ${libopskernel} ${libaicore_utils} ${libaicpu_engine_common} ${acl})
endif()
