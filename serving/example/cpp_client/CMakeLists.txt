cmake_minimum_required(VERSION 3.5.1)

project(MSClient C CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(Threads REQUIRED)

# This branch assumes that gRPC and all its dependencies are already installed
# on this system, so they can be located by find_package().

# Find Protobuf installation
# Looks for protobuf-config.cmake file installed by Protobuf's cmake installation.
option(GRPC_PATH "set grpc path")
if(GRPC_PATH)
  set(CMAKE_PREFIX_PATH ${GRPC_PATH})
  set(protobuf_MODULE_COMPATIBLE TRUE)
  find_package(Protobuf CONFIG REQUIRED)
  message(STATUS "Using protobuf ${protobuf_VERSION}, CMAKE_PREFIX_PATH : ${CMAKE_PREFIX_PATH}")
elseif(NOT GRPC_PATH AND grpc_ROOT)
  add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
  if (EXISTS ${grpc_ROOT}/lib64)
    set(gRPC_DIR "${grpc_ROOT}/lib64/cmake/grpc")
  elseif(EXISTS ${grpc_ROOT}/lib)
    set(gRPC_DIR "${grpc_ROOT}/lib/cmake/grpc")
  endif()
  add_library(protobuf::libprotobuf ALIAS protobuf::protobuf)
  add_executable(protobuf::libprotoc ALIAS protobuf::protoc)
  message(STATUS "serving using grpc_DIR : " ${gRPC_DIR})
elseif(NOT gRPC_DIR AND NOT GRPC_PATH)
  message(FATAL_ERROR "please check gRPC. If the client is compiled separately，you can use the command: cmake -D GRPC_PATH=xxx\n" "XXX is the gRPC installation path")
endif()

if(CMAKE_CROSSCOMPILING)
  find_program(_PROTOBUF_PROTOC protoc)
else()
  set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)
endif()

# Find gRPC installation
# Looks for gRPCConfig.cmake file installed by gRPC's cmake installation.
find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")

if(CMAKE_CROSSCOMPILING)
  find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
else()
  set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
endif()

# Proto file
get_filename_component(hw_proto "../../ms_service.proto" ABSOLUTE)
get_filename_component(hw_proto_path "${hw_proto}" PATH)

# Generated sources
set(hw_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/ms_service.pb.cc")
set(hw_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/ms_service.pb.h")
set(hw_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/ms_service.grpc.pb.cc")
set(hw_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/ms_service.grpc.pb.h")
add_custom_command(
        OUTPUT "${hw_proto_srcs}" "${hw_proto_hdrs}" "${hw_grpc_srcs}" "${hw_grpc_hdrs}"
        COMMAND ${_PROTOBUF_PROTOC}
        ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${hw_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${hw_proto}"
        DEPENDS "${hw_proto}")

# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

# Targets greeter_[async_](client|server)
add_executable(ms_client "ms_client.cc"
        ${hw_proto_srcs}
        ${hw_grpc_srcs})
target_link_libraries(ms_client
        gRPC::grpc++_reflection
        gRPC::grpc++
        protobuf::libprotobuf)
