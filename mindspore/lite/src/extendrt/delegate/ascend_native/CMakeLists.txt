add_library(ascend_native_mid OBJECT
        ${CMAKE_CURRENT_SOURCE_DIR}/delegate.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/delegate_allocator.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/sub_graph_helper.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/stub_kernel.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ascend_native_plugin_impl.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/ascend_native_composite.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/ascend_native_stub.cc
        )

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ascend_native_impl)
