add_library(hccl_plugin SHARED hccl_plugin.cc)
target_include_directories(hccl_plugin PRIVATE ${CMAKE_BINARY_DIR}/proto/ge)
add_dependencies(hccl_plugin ge_proto)

find_library(HCCL hccl ${ASCEND_CANN_RUNTIME_PATH})
target_link_libraries(hccl_plugin -Wl,--no-as-needed ${HCCL})
