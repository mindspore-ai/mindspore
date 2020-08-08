find_package(Threads REQUIRED)

# This branch assumes that gRPC and all its dependencies are already installed
# on this system, so they can be located by find_package().

# Find Protobuf installation
# Looks for protobuf-config.cmake file installed by Protobuf's cmake installation.

#set(protobuf_MODULE_COMPATIBLE TRUE)
#find_package(Protobuf CONFIG REQUIRED)
#message(STATUS "Using protobuf ${protobuf_VERSION}")
add_library(protobuf::libprotobuf ALIAS protobuf::protobuf)
add_executable(protobuf::libprotoc ALIAS protobuf::protoc)

set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
if (CMAKE_CROSSCOMPILING)
    find_program(_PROTOBUF_PROTOC protoc)
else ()
    set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)
endif ()

# Find gRPC installation
# Looks for gRPCConfig.cmake file installed by gRPC's cmake installation.
if (EXISTS ${grpc_ROOT}/lib64)
    set(gRPC_DIR "${grpc_ROOT}/lib64/cmake/grpc")
else ()
    set(gRPC_DIR "${grpc_ROOT}/lib/cmake/grpc")
endif ()
message("serving ut using grpc_DIR : " ${gPRC_DIR})

find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")

set(_GRPC_GRPCPP gRPC::grpc++)
set(_REFLECTION gRPC::grpc++_reflection)

if (CMAKE_CROSSCOMPILING)
    find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
    find_program(_GRPC_PYTHON_PLUGIN_EXECUTABLE grpc_python_plugin)
else ()
    set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
    set(_GRPC_PYTHON_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_python_plugin>)
endif ()

# Proto file
get_filename_component(hw_proto "ms_service.proto" ABSOLUTE)
get_filename_component(hw_proto_path ${hw_proto} PATH)
# Generated sources
set(hw_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/ms_service.pb.cc")
set(hw_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/ms_service.pb.h")
set(hw_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/ms_service.grpc.pb.cc")
set(hw_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/ms_service.grpc.pb.h")
set(hw_py_pb2 "${CMAKE_CURRENT_BINARY_DIR}/ms_service_pb2.py")
set(hw_py_pb2_grpc "${CMAKE_CURRENT_BINARY_DIR}/ms_service_pb2_grpc.py")
add_custom_command(
        OUTPUT "${hw_proto_srcs}" "${hw_proto_hdrs}" "${hw_grpc_srcs}" "${hw_grpc_hdrs}" "${hw_py_pb2}" "${hw_py_pb2_grpc}"
        COMMAND ${_PROTOBUF_PROTOC}
        ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${hw_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${hw_proto}"
        COMMAND ${_PROTOBUF_PROTOC}
        ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --python_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${hw_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_PYTHON_PLUGIN_EXECUTABLE}"
        "${hw_proto}"
        DEPENDS "${hw_proto}")

list(APPEND SERVING_SRC_TEST ${hw_proto_srcs} ${hw_grpc_srcs})

file(GLOB_RECURSE ACL_SESSION_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "../../../../serving/acl/*.cc"
        "../../../../serving/core/*.cc")
list(APPEND SERVING_SRC_TEST ${ACL_SESSION_SRC_LIST})

# utest files
file(GLOB_RECURSE ACL_UTEST_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
list(APPEND SERVING_SRC_TEST ${ACL_UTEST_SRC_LIST})

include_directories(${CMAKE_SOURCE_DIR}/serving/core)
include_directories(${CMAKE_SOURCE_DIR}/serving/acl)
include_directories(${CMAKE_SOURCE_DIR}/serving)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../)
add_library(ut_serving_obj OBJECT ${SERVING_SRC_TEST})


