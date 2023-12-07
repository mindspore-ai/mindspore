if(DEFINED ENV{MSLITE_ASCEND_TARGET})
set(MSLITE_ASCEND_TARGET $ENV{MSLITE_ASCEND_TARGET})
endif()
if(DEFINED ENV{ASCEND_CUSTOM_PATH})
set(ASCEND_CUSTOM_PATH $ENV{ASCEND_CUSTOM_PATH})
endif()
set(ASCEND_PRODUCT_TYPE ${MSLITE_ASCEND_TARGET})
set(ASCEND_INSTALL_PATH ${ASCEND_PATH}/latest)

set(ASCEND_RUN_MODE "ONBOARD")
list(APPEND CMAKE_MODULE_PATH ${CCE_CMAKE_PATH})
add_library(ascend_native_mid OBJECT
        ${CMAKE_CURRENT_SOURCE_DIR}/delegate.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/delegate_allocator.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/sub_graph_helper.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/stub_kernel.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ascend_native_plugin_impl.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/ascend_native_composite.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/ascend_native_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ascend_native_registration_factory.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ascend_native_matmul_kernel.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ascend_native_encoder_kernel.cc
        )
target_include_directories(ascend_native_mid PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ascend_native_impl)
target_compile_options(ascend_native_mid PUBLIC -std=c++17)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ascend_native_impl)
