cmake_minimum_required(VERSION 3.14)
project(QuickStartCpp)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.3.0)
    message(FATAL_ERROR "GCC version ${CMAKE_CXX_COMPILER_VERSION} must not be less than 7.3.0")
endif()

if(DEFINED ENV{LITE_HOME})
    set(LITE_HOME $ENV{LITE_HOME})
endif()

if(DEFINED ENV{ASCEND_HOME})
    set(ASCEND_HOME $ENV{ASCEND_HOME})
endif()

# Add directory to include search path
include_directories(${LITE_HOME}/runtime)
# Add directory to linker search path
link_directories(${LITE_HOME}/runtime/lib)
link_directories(${LITE_HOME}/tools/converter/lib)

set(ASCEND_CANN_RUNTIME_PATH ${ASCEND_HOME}/latest/x86_64-linux/lib64)
find_library(ge_graph libgraph.so ${ASCEND_CANN_RUNTIME_PATH})
find_library(acl libascendcl.so ${ASCEND_CANN_RUNTIME_PATH})
find_library(acl_cblas libacl_cblas.so ${ASCEND_CANN_RUNTIME_PATH})
find_library(acl_dvpp libacl_dvpp.so ${ASCEND_CANN_RUNTIME_PATH})
find_library(acl_runtime libruntime.so ${ASCEND_CANN_RUNTIME_PATH})
find_library(ge_compiler libge_compiler.so ${ASCEND_CANN_RUNTIME_PATH})
find_library(libplatform libplatform.so ${ASCEND_CANN_RUNTIME_PATH})
find_library(libcompress libcompress.so ${ASCEND_CANN_RUNTIME_PATH})
find_library(libopskernel libopskernel.so ${ASCEND_CANN_RUNTIME_PATH})
find_library(libaicore_utils libaicore_utils.so ${ASCEND_CANN_RUNTIME_PATH})
find_library(libaicpu_engine_common libaicpu_engine_common.so ${ASCEND_CANN_RUNTIME_PATH})
find_library(ge_runner libge_runner.so ${ASCEND_CANN_RUNTIME_PATH})

file(GLOB_RECURSE ASCEND_GE_DISTRIBUTED ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
add_executable(ascend_ge_distributed ${ASCEND_GE_DISTRIBUTED})

target_link_libraries(ascend_ge_distributed ${ge_graph} ${ge_compiler} ${acl_retr} ${acl_cblas} ${acl_dvpp}
                      ${acl_runtime} ${libplatform} ${libcompress} ${libopskernel} ${libaicore_utils}
                      ${libaicpu_engine_common} ${acl} ${ge_runner} mindspore-lite)


