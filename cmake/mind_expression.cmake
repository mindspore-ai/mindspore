set(SECURE_CXX_FLAGS "")
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    if(WIN32)
        set(SECURE_CXX_FLAGS "-fstack-protector-all")
    else()
    set(SECURE_CXX_FLAGS "-fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack")
    endif()
endif()
set(_ms_tmp_CMAKE_CXX_FLAGS_F ${CMAKE_CXX_FLAGS})

if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
endif()

set(TOP_DIR ${CMAKE_SOURCE_DIR})

include(cmake/utils.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/external_libs/robin.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/external_libs/eigen.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/external_libs/json.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/dependency_securec.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/external_libs/protobuf.cmake)

if(MS_BUILD_GRPC)
    # build dependencies of gRPC
    include(${CMAKE_SOURCE_DIR}/cmake/external_libs/openssl.cmake)
    include(${CMAKE_SOURCE_DIR}/cmake/external_libs/absl.cmake)
    include(${CMAKE_SOURCE_DIR}/cmake/external_libs/c-ares.cmake)
    include(${CMAKE_SOURCE_DIR}/cmake/external_libs/zlib.cmake)
    include(${CMAKE_SOURCE_DIR}/cmake/external_libs/re2.cmake)
    # build gRPC
    include(${CMAKE_SOURCE_DIR}/cmake/external_libs/grpc.cmake)
    # build event
    include(${CMAKE_SOURCE_DIR}/cmake/external_libs/libevent.cmake)
endif()

include(${CMAKE_SOURCE_DIR}/cmake/external_libs/pybind11.cmake)
MESSAGE("go to link flatbuffers")
include(${CMAKE_SOURCE_DIR}/cmake/external_libs/flatbuffers.cmake)
if(USE_GLOG)
    include(${CMAKE_SOURCE_DIR}/cmake/external_libs/glog.cmake)
endif()

find_package(Python3)
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/third_party)
if(ENABLE_MPI)
    include(${CMAKE_SOURCE_DIR}/cmake/external_libs/ompi.cmake)
endif()

if(ENABLE_CPU)
    include(${CMAKE_SOURCE_DIR}/cmake/external_libs/mkl_dnn.cmake)
endif()

if(MSVC)
    include(${CMAKE_SOURCE_DIR}/cmake/external_libs/dirent.cmake)
endif()

if(ENABLE_GPU AND GPU_BACKEND_CUDA)
    include(${CMAKE_SOURCE_DIR}/cmake/external_libs/cub.cmake)
    include(${CMAKE_SOURCE_DIR}/cmake/external_libs/cucollections.cmake)
    if(ENABLE_MPI)
        include(${CMAKE_SOURCE_DIR}/cmake/external_libs/nccl.cmake)
    endif()
endif()

if(ENABLE_MINDDATA)
    include(${CMAKE_SOURCE_DIR}/cmake/external_libs/icu4c.cmake)
    include(${CMAKE_SOURCE_DIR}/cmake/external_libs/opencv.cmake)
    include(${CMAKE_SOURCE_DIR}/cmake/external_libs/sqlite.cmake)
    include(${CMAKE_SOURCE_DIR}/cmake/external_libs/tinyxml2.cmake)
    include(${CMAKE_SOURCE_DIR}/cmake/external_libs/cppjieba.cmake)
    include(${CMAKE_SOURCE_DIR}/cmake/external_libs/sentencepiece.cmake)
endif()

if(ENABLE_MINDDATA)
    include(${CMAKE_SOURCE_DIR}/cmake/external_libs/jpeg_turbo.cmake)
endif()

if(ENABLE_TESTCASES OR ENABLE_CPP_ST)
    include(${CMAKE_SOURCE_DIR}/cmake/external_libs/gtest.cmake)
endif()

set(CMAKE_CXX_FLAGS ${_ms_tmp_CMAKE_CXX_FLAGS_F})
