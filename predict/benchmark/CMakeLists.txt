
cmake_minimum_required(VERSION 3.12)
project(benchmark)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "Debug")

#include 3rd
include_directories(${3RD_DIR}/protobuf/build/include)
include_directories(${3RD_DIR}/securec/include)
include_directories(${3RD_DIR}/flatbuffers/include)
include_directories(${3RD_DIR}/googletest/googletest/include)
include_directories(${3RD_DIR}/googletest/googlemock/include)
include_directories(${PREDICT_DIR}/module/tvm_kernel/incubator-tvm/3rdparty/dlpack/include)
include_directories(${3RD_DIR}/flatbuffers/include)
include_directories(${3RD_DIR}/securec/include)

#include ms
include_directories(.)
include_directories(${PREDICT_DIR})

set(COMMON_SRC ${PREDICT_DIR}/common/flag_parser.cc
	       ${PREDICT_DIR}/common/file_utils.cc
	       ${PREDICT_DIR}/common/func_utils.cc
	       ${PREDICT_DIR}/common/mslog.cc
	       ${PREDICT_DIR}/common/utils.cc)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../output/lib/)

add_executable(benchmark main.cc benchmark.cc ${COMMON_SRC})

target_link_libraries(benchmark mspredict libsecurec.a)
add_dependencies(benchmark tvm_kernel)
add_dependencies(benchmark securec)

add_custom_command(TARGET benchmark POST_BUILD
        COMMAND mkdir -pv ${DOTEST_DIR}
        COMMAND cp ${PREDICT_BUILD_DIR}/benchmark/benchmark ${DOTEST_DIR})
