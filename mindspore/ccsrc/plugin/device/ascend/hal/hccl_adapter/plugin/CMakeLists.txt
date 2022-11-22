add_library(hccl_plugin SHARED hccl_plugin.cc)
target_include_directories(hccl_plugin PRIVATE ${CMAKE_BINARY_DIR}/proto/ge)
add_dependencies(hccl_plugin graph)

set(HCCL_RPATH ${ASCEND_RPATH})
set(HCCL_RPATH ${HCCL_RPATH}:/usr/local/Ascend/nnae/latest/lib64/plugin/opskernel)
set(HCCL_RPATH ${HCCL_RPATH}:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/opskernel)
set(HCCL_RPATH ${HCCL_RPATH}:/usr/local/Ascend/latest/lib64/plugin/opskernel)

set_target_properties(hccl_plugin PROPERTIES INSTALL_RPATH ${HCCL_RPATH})

find_library(HCCL hccl ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
find_library(REGISTER register ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
find_library(HCCL_ADPTER hcom_graph_adaptor ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
find_library(HCCL_RA ra ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
find_library(HCCL_BUILDER hcom_opskernel_builder ${ASCEND_CANN_PLUGIN_PATH} ${ASCEND_TOOLKIT_PLUGIN_PATH})
target_link_libraries(hccl_plugin -Wl,--no-as-needed ${HCCL} ${HCCL_ADPTER} ${REGISTER} ${HCCL_BUILDER} ${HCCL_RA})
