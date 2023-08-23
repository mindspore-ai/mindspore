cmake_minimum_required(VERSION 3.12)
project(NNIE_Custom)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
if(${MSLITE_REGISTRY_DEVICE}  STREQUAL "Hi3516D")
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_patry/hi3516_sdk/)
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_patry/hi3516_sdk/lib)
elseif(${MSLITE_REGISTRY_DEVICE}  STREQUAL "Hi3519A")
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_patry/hi3519_sdk/)
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_patry/hi3519_sdk/lib)
elseif(${MSLITE_REGISTRY_DEVICE}  STREQUAL "Hi3559A")
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_patry/hi3559_sdk/)
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_patry/hi3559_sdk/lib)
endif()
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../runtime)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../runtime/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../runtime/include/third_party)


aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src COMMON_SRC3)

set(MSLITE_NNIE_LINK_LIB nnie mpi VoiceEngine upvqe dnvqe)

add_library(mslite_nnie SHARED
        ${COMMON_SRC3}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/litert/cxx_api/kernel.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/registry/kernel_interface_registry.cc)
target_link_libraries(mslite_nnie ${MSLITE_NNIE_LINK_LIB} securec)

