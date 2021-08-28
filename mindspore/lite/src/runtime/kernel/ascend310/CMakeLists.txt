include_directories(${TOP_DIR}/graphengine/inc/external)

find_library(ge_graph libgraph.so ${ASCEND_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})

aux_source_directory(src ACL_SRC)
add_library(ascend310_kernel_mid OBJECT ${ACL_SRC})

add_dependencies(ascend310_kernel_mid fbs_inner_src)

target_link_libraries(ascend310_kernel_mid ${ge_graph} ${ge_compiler}
        ${acl_retr} ${acl_cblas} ${acl_dvpp} ${acl_runtime} ${libplatform}
        ${libcompress} ${libopskernel} ${libaicore_utils} ${libaicpu_engine_common} ${acl})
