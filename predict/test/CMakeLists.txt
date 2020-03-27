cmake_minimum_required(VERSION 3.12)
project(ms-test)

set(CMAKE_CXX_STANDARD 11)

#include 3rd
include_directories(${3RD_DIR}/securec/include)
include_directories(${3RD_DIR}/flatbuffers/include)
include_directories(${3RD_DIR}/googletest/googletest/include)
include_directories(${3RD_DIR}/googletest/googlemock/include)
include_directories(${3RD_DIR}/securec/include)

#include ms
include_directories(.)
include_directories(..)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../output/lib/)

set(COMMON_SRC ${PREDICT_DIR}/common/flag_parser.cc
	       ${PREDICT_DIR}/common/file_utils.cc
	       ${PREDICT_DIR}/common/mslog.cc
	       ${PREDICT_DIR}/common/storage.cc
	       ${PREDICT_DIR}/common/utils.cc)

#tools src
file(GLOB_RECURSE TOOLS_SRC ../tools/*.cpp)

add_executable(ms-test
	${COMMON_SRC}
        ${TOOLS_SRC}
        src/graph_tests.cc
        benchmark/benchmark_tests.cc
        ${CMAKE_SOURCE_DIR}/benchmark/benchmark.cc
        ${TF_PROTO_SRC}
        ${MS_CONVERTER_SRC}
        test_context.h
        test_context.cc
        main.cc)

target_link_libraries(ms-test mspredict gtest libsecurec.a)
add_dependencies(ms-test securec)
add_dependencies(ms-test gtest)

# copy test file
add_custom_command(TARGET ms-test POST_BUILD
        COMMAND mkdir -pv ${DOTEST_DIR}
        COMMAND cp ${PREDICT_BUILD_DIR}/test/ms-test ${DOTEST_DIR}
        COMMAND cp ${PREDICT_DIR}/test/run_tests.sh ${PREDICT_BUILD_DIR}/test/
        COMMAND cp -r ${PREDICT_DIR}/test/data/ ${PREDICT_BUILD_DIR}/test/doTest/)
