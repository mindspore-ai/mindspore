cmake_minimum_required(VERSION 3.14)
project(DPICO_CONVERTER)

__download_pkg(pico_mapper
        http://mindspore-repo.csi.rnd.huawei.com/mindspore/enterprise/dpico/pico_mapper_1115.tar.gz
        b7c8ea7f4585cbc629b67a4a6ac1ad98)
__download_pkg(protobuf-3.9-for-dpico
        http://mindspore-repo.csi.rnd.huawei.com/mindspore/enterprise/dpico/protobuf-3.9.0.tar.gz
        b5c5a0ba33d301ae51a2f878c17e88be)
__download_pkg(opencv-4.2-for-dpico
        http://mindspore-repo.csi.rnd.huawei.com/mindspore/enterprise/dpico/opencv-4.2.0.tar.gz
        00d2ff839f74c0314ee28acb120fff9a)
__download_pkg(proto-for-dpico
        http://mindspore-repo.csi.rnd.huawei.com/mindspore/enterprise/dpico/proto1030.tar.gz
        461563613d3205045ca3008fd43aea63)

set(PROTOC ${protobuf-3.9-for-dpico_SOURCE_DIR}/bin/protoc)
execute_process(COMMAND
        ${PROTOC} -I=${proto-for-dpico_SOURCE_DIR} --cpp_out=${proto-for-dpico_SOURCE_DIR} pico_caffe.proto)
execute_process(COMMAND
        ${PROTOC} -I=${TOP_DIR}/third_party/proto/onnx --cpp_out=${proto-for-dpico_SOURCE_DIR} onnx.proto)

file(GLOB_RECURSE DPICO_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/parser/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/checker/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/infer/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/mapper/*.cc
        )

set(PICO_MAPPER_LIB_PATH ${pico_mapper_SOURCE_DIR}/lib CACHE STRING "dpico mapper lib path")
set(OPENCV_4.2.0_LIB_PATH ${opencv-4.2-for-dpico_SOURCE_DIR}/lib CACHE STRING "dpico opencv-4.2.0 lib path")
set(PROTOBUF_3.9.0_LIB_PATH ${protobuf-3.9-for-dpico_SOURCE_DIR}/lib CACHE STRING "dpico protobuf-3.9.0 lib path")

include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(BEFORE ${opencv-4.2-for-dpico_SOURCE_DIR}/include/opencv4)
include_directories(BEFORE ${protobuf-3.9-for-dpico_SOURCE_DIR}/include)
include_directories(${pico_mapper_SOURCE_DIR}/include)
include_directories(${proto-for-dpico_SOURCE_DIR})

link_directories(${pico_mapper_SOURCE_DIR}/lib)
set_property(SOURCE ${DPICO_SRC} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)

add_library(dpico_atc_adapter SHARED ${DPICO_SRC})
add_dependencies(dpico_atc_adapter fbs_src fbs_inner_src)
target_link_libraries(dpico_atc_adapter
        pico_mapper_api
        ${PROTOBUF_3.9.0_LIB_PATH}/libprotobuf.so.3.9.0.0
        ${OPENCV_4.2.0_LIB_PATH}/libopencv_core.so.4.2
        ${OPENCV_4.2.0_LIB_PATH}/libopencv_imgcodecs.so.4.2
        ${OPENCV_4.2.0_LIB_PATH}/libopencv_imgproc.so.4.2
        stdc++fs)

if((NOT WIN32) AND ${CMAKE_BUILD_TYPE} STREQUAL "Release")
    add_custom_command(TARGET dpico_atc_adapter POST_BUILD COMMAND strip
            ${CMAKE_CURRENT_BINARY_DIR}/libdpico_atc_adapter.so)
endif()
