set(ENABLE_CPU on)
add_definitions(-DUSE_GLOG)
string(REPLACE "-fno-rtti" "" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
string(REPLACE "-fno-rtti" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
set(MINDSPORE_PROJECT_DIR ${TOP_DIR})

set(SERVER_FLATBUFFER_OUTPUT "${CMAKE_BINARY_DIR}/schema")
set(FBS_FILES
        ${TOP_DIR}/mindspore/schema/cipher.fbs
        ${TOP_DIR}/mindspore/schema/fl_job.fbs
        )
ms_build_flatbuffers(FBS_FILES ${CMAKE_CURRENT_SOURCE_DIR}../../schema generated_fbs_files ${SERVER_FLATBUFFER_OUTPUT})

file(GLOB_RECURSE COMM_PROTO_IN RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${CCSRC_DIR}/ps/core/protos/*.proto")
message(${COMM_PROTO_IN})
ms_protobuf_generate(COMM_PROTO_SRCS COMM_PROTO_HDRS ${COMM_PROTO_IN})
list(APPEND MINDSPORE_PROTO_LIST ${COMM_PROTO_SRCS})

include(${TOP_DIR}/cmake/external_libs/robin.cmake)
include(${TOP_DIR}/cmake/external_libs/eigen.cmake)
include(${TOP_DIR}/cmake/external_libs/mkl_dnn.cmake)

find_package(Python3 COMPONENTS Interpreter Development)
if(Python3_FOUND)
  find_package(Python3 COMPONENTS NumPy Development)

  if(Python3_NumPy_FOUND)
    include_directories(${Python3_INCLUDE_DIRS})
    include_directories(${Python3_NumPy_INCLUDE_DIRS})
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../core/)
    include(${TOP_DIR}/cmake/external_libs/pybind11.cmake)
  endif()
endif()
include(${TOP_DIR}/cmake/external_libs/libevent.cmake)

# add_subdirectory(${CCSRC_DIR}/plugin/device/cpu/kernel mindspore_ccsrc_plugin_device_cpu_kernel)
add_subdirectory(${CCSRC_DIR}/backend/common/session mindspore_ccsrc_backend_common_session)