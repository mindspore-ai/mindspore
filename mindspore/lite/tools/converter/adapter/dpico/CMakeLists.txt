cmake_minimum_required(VERSION 3.14)
project(DPICO_CONVERTER)

if(DEFINED ENV{HISI_SDK_PATH})
    set(SD3403_ATC_LIB_PATH $ENV{HISI_SDK_PATH}/sd3403_sdk/dpico_atc_adapter)
else()
    message(FATAL_ERROR "error, HISI_SDK_PATH env not found")
endif()

set(OPENCV_PATH ${SD3403_ATC_LIB_PATH}/opencv-4.5.2)
set(PROTOBUF_PATH ${SD3403_ATC_LIB_PATH}/protobuf-3.13.0)
set(PICO_PROTO_PATH ${SD3403_ATC_LIB_PATH}/pico_proto_0924)
set(PICO_MAPPER_PATH ${SD3403_ATC_LIB_PATH}/pico_mapper_0924)

set(PROTOC ${PROTOBUF_PATH}/bin/protoc)
set(PROTO_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/pico_proto)
file(MAKE_DIRECTORY ${PROTO_OUTPUT_PATH})
execute_process(COMMAND chmod +x ${PROTOC})
execute_process(COMMAND
        ${PROTOC} -I=${PICO_PROTO_PATH} --cpp_out=${PROTO_OUTPUT_PATH} pico_caffe.proto)
execute_process(COMMAND
        ${PROTOC} -I=${TOP_DIR}/third_party/proto/onnx --cpp_out=${PROTO_OUTPUT_PATH} onnx.proto)

file(GLOB_RECURSE DPICO_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/parser/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/checker/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/infer/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/mapper/*.cc
        ${TOOLS_DIR}/converter/adapter/utils.cc
        ${PROTO_OUTPUT_PATH}/onnx.pb.cc
        ${PROTO_OUTPUT_PATH}/pico_caffe.pb.cc
        )

include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(BEFORE ${OPENCV_PATH}include/opencv4)
include_directories(BEFORE ${PROTOBUF_PATH}/include)
include_directories(${PICO_MAPPER_PATH}/include)
include_directories(${PROTO_OUTPUT_PATH})
include_directories(${TOOLS_DIR}/converter)

link_directories(${pico_mapper_SOURCE_DIR}/lib)
set_property(SOURCE ${DPICO_SRC} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)

add_library(dpico_atc_adapter SHARED ${DPICO_SRC})
add_dependencies(dpico_atc_adapter fbs_src fbs_inner_src)
target_link_libraries(dpico_atc_adapter
        ${PICO_MAPPER_PATH}/lib/libsvp_atc_api.so
        ${PROTOBUF_PATH}/lib/libprotobuf.so.3.13.0.0
        ${OPENCV_PATH}/lib/libopencv_core.so.4.5.2
        ${OPENCV_PATH}/lib/libopencv_imgcodecs.so.4.5.2
        ${OPENCV_PATH}/lib/libopencv_imgproc.so.4.5.2
        stdc++fs)

if((NOT WIN32) AND ${CMAKE_BUILD_TYPE} STREQUAL "Release")
    add_custom_command(TARGET dpico_atc_adapter POST_BUILD COMMAND strip
            ${CMAKE_CURRENT_BINARY_DIR}/libdpico_atc_adapter.so)
endif()
