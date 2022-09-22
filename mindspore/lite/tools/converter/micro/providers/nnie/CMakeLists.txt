cmake_minimum_required(VERSION 3.14)
project(micro_nnie_kernel)
if(NOT DEFINED PKG_PATH)
    message(FATAL_ERROR "PKG_PATH not set")
endif()
message("PKG_PATH:${PKG_PATH}")
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../)
set(THIRD_PATRY_PATH ${LITE_DIR}/providers/nnie/third_patry/)
if(${MSLITE_REGISTRY_DEVICE}  STREQUAL "Hi3516D")
    include_directories(${THIRD_PATRY_PATH}/hi3516_sdk/)
    link_directories(${THIRD_PATRY_PATH}/hi3516_sdk/lib)
elseif(${MSLITE_REGISTRY_DEVICE}  STREQUAL "Hi3519A")
    include_directories(${THIRD_PATRY_PATH}/hi3519_sdk/)
    link_directories(${THIRD_PATRY_PATH}/hi3519_sdk/lib)
elseif(${MSLITE_REGISTRY_DEVICE}  STREQUAL "Hi3559A")
    include_directories(${THIRD_PATRY_PATH}/hi3559_sdk/)
    link_directories(${THIRD_PATRY_PATH}/hi3559_sdk/lib)
endif()

include_directories(${PKG_PATH}/tools/codegen/include)
include_directories(${PKG_PATH}/runtime/)
include_directories(${LITE_DIR}/providers/nnie)
include_directories(${LITE_DIR}/providers/nnie_proposal)

set(SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/nnie_micro.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/nnie_interfaces.cc
  ${LITE_DIR}/providers/nnie/src/nnie_common.cc
  ${LITE_DIR}/providers/nnie/src/nnie_memory.cc
  ${LITE_DIR}/providers/nnie/src/nnie_print.cc
  ${LITE_DIR}/providers/nnie_proposal/src/proposal.cc)

add_library(micro_nnie SHARED ${SRC})
target_link_libraries(micro_nnie nnie mpi VoiceEngine upvqe dnvqe securec)
